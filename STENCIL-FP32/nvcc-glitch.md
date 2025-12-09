# README — Illegal Instruction (CUDA) when running jacobi2D_cudaDMA

**Short summary**

When compiling and running the `jacobi2D_cudaDMA` program in release/optimized builds, the kernel launch fails at runtime with:

```
CUDA Error after kernel launch: an illegal instruction was encountered
```

However, when compiled with debug/device-debug flags (`-g -G`), the program runs without producing the illegal-instruction error. This README documents reproduction steps, likely cause, diagnostics performed, and recommended fixes / workarounds.

---

## Files

- Kernel/source: `./jacobi2D_cudaDMA.cu`
- Header / DMA helper: `./jacobi2D_cudaDMA.cuh`
- Makefile used for builds: `./Makefile_cudaDMA`

(Direct file paths above can be used to inspect the exact implementation and the DMA helper. If your environment supports mapping these paths to a URL, point your file browser to those paths.)

---

## Observed behavior

1. **Debug build** (flags: `-g -G`, optionally `-O0`) — Program runs successfully; no illegal-instruction error.
2. **Optimized / Release build** (usual `-O2`/`-O3`, without `-G`) — Kernel launch returns `CUDA` / `cudaErrorIllegalInstruction` after launch, usually around the call to `start_async_dma()` (or a nearby DMA helper call).

This pattern strongly suggests the compiled device code in the optimized build contains GPU instructions the actual hardware cannot execute.

### Additional suspicious behavior (kernel launch uncertainty)

**Critical observation:** When running with `-g -G` flags (debug build):

- No illegal instruction error occurs
- **No CPU-GPU mismatch reported** (results appear correct)
- **BUT: Print statements inside the kernel are NOT printed to console**

This raises a concerning possibility: **The kernel may not be launching at all in debug mode**, yet the program reports no errors and shows no CPU-GPU mismatch.

**Possible explanations:**

1. **Silent kernel launch failure with undetected error:**

   - The kernel launch may be failing silently, and the error is not being checked immediately after launch
   - Need to add explicit `cudaGetLastError()` and `cudaDeviceSynchronize()` checks after kernel launch

2. **Output buffer never modified (remains initialized):**

   - If the output buffer is pre-initialized with values that match the expected CPU result (e.g., zeros or correct values from a previous run), the comparison would falsely pass even if the kernel never ran
   - The debug build might be skipping kernel execution due to compiler optimization or launch configuration issues

3. **Print statements suppressed or buffered:**
   - Device-side `printf` may require explicit flushing or may not work correctly with certain cudaDMA synchronization patterns
   - The debug build's synchronization behavior might prevent printf buffer from being flushed to host

**Recommended diagnostic steps:**

1. **Verify kernel actually launches:**

   ```cpp
   kernel<<<grid, block>>>(args);
   cudaError_t launch_err = cudaGetLastError();
   if (launch_err != cudaSuccess) {
       printf("Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
   }
   cudaError_t sync_err = cudaDeviceSynchronize();
   if (sync_err != cudaSuccess) {
       printf("Kernel execution failed: %s\n", cudaGetErrorString(sync_err));
   }
   ```

2. **Add sentinel values to output buffer:**

   - Initialize output buffer with distinctive sentinel values (e.g., `-999.0f`) before kernel launch
   - Check if these values change after kernel execution
   - If sentinels remain unchanged, kernel definitely did not execute

3. **Use CUDA events to measure execution time:**

   ```cpp
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   kernel<<<grid, block>>>(args);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float ms;
   cudaEventElapsedTime(&ms, start, stop);
   printf("Kernel execution time: %f ms\n", ms);
   ```

   - If execution time is near-zero (< 0.001 ms), kernel likely didn't run

4. **Check printf output explicitly:**
   - Add `cudaDeviceSynchronize()` after kernel launch to ensure printf buffer is flushed
   - Try using `printf("KERNEL ENTRY\\n");` as the very first line in kernel
   - Check both stdout and stderr, and redirect output: `./program 2>&1 | tee output.log`

---

## Likely root causes

1. **Architecture-specific instructions emitted in optimized code**

   - The DMA helper (`CudaDMAStrided`) likely emits older GPU instructions which are no loger supported or recognized in newer GPUs.
   - In optimized builds, NVCC can generate those instructions (or inline them) if the compile target allows it. If the runtime GPU does not support them, the kernel will raise an illegal instruction at runtime.

2. **Optimizations exposing architecture-dependent code paths**

   - With optimizations enabled, the compiler may inline or transform code in ways that cause an unsupported instruction to appear unguarded or executed where the fallback was expected.

3. **Missing `__CUDA_ARCH__` guards**

   - The device-side DMA helper may not be fully guarded with `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= XXX)` checks for the advanced instructions, or the guards are not preventing emission in optimized builds.

---

## Diagnostics performed (recommended / done)

- Verified behavior difference between `-g -G` and non-debug builds.
- Reproduced crash location near `start_async_dma()` in the kernel.
- **Observed that print statements inside kernel do NOT appear in console output, even in debug builds that report "correct" results**
- **Suspicion: Kernel may not be executing at all, but falsely reporting success due to uninitialized/pre-initialized output buffers**
- Suggested tools to pinpoint exact failing instruction (use these if not yet run):

  - `cuda-memcheck ./a.out` or `cuda-memcheck --tool=memcheck ./a.out` to catch illegal instruction events.
  - `cuda-gdb` breakpoints inside `jacobi2D_kernel_pure_cudaDMA` and inside `start_async_dma()` to step into device code and inspect the PC/SASS.
  - `cuobjdump --dump-sass a.out` (or `nvdisasm`) to inspect the emitted SASS/ptx and find unsupported opcodes.
  - `deviceQuery` / `nvidia-smi` to confirm GPU model and compute capability.
  - **Add explicit error checking after kernel launch** (see diagnostic steps in "Observed behavior" section above)
  - **Initialize output buffer with sentinel values** to verify kernel execution
  - **Use CUDA events to measure kernel execution time** to confirm kernel runs

---

## Immediate workaround

- Compile with device debugging turned on to avoid the illegal instruction: add `-G` (and `-g` for host debug symbols) to nvcc. Example:

```bash
nvcc -G -g -O0 -arch=sm_70 -o jacobi_debug jacobi2D_cudaDMA.cu
```

**Note:** `-G` is a debugging-only option — it disables many device optimizations and replaces advanced instructions with conservative code sequences. It is not suitable for production/performance runs; it is a diagnostic / temporary workaround.

---

## Recommended permanent fixes

1. **Guard advanced device instructions by compute capability**

Inside `CudaDMAStrided` and any helper that emits inline PTX or architecture-specific code, ensure you guard fast/advanced paths with compile-time checks. For example:

```cpp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  // Ampere+: emit cp.async or other hw accelerated ops
#else
  // Portable fallback: normal loads/stores or thread-based copy
#endif
```

2. **Provide a software fallback that is always valid**

- Implement a fallback copy path (simple `ld.global`/`st.shared` via regular loads or a thread-copy loop) that will be used on older SMs.

3. **Target correct architecture(s) when compiling**

- Ensure nvcc `-gencode` / `-arch` flags match the GPU(s) you will run on. Compiling for a newer arch than your runtime GPU can lead to emitted instructions that the hardware doesn't support.

Example targeting both a safe minimum and an Ampere fast path:

```bash
nvcc -gencode=arch=compute_70,code=sm_70 \
     -gencode=arch=compute_80,code=sm_80 \
     -O3 -o jacobi_release jacobi2D_cudaDMA.cu
```

4. **If using inline PTX or asm, guard and test thoroughly**

- Inline PTX must be guarded and tested on each SM level. Prefer `asm volatile` only inside `#if __CUDA_ARCH__` conditionals.

5. **Run static inspection and disassembly when suspicious**

- Use `cuobjdump --dump-sass` and search for `cp.async` (or other opcodes) in the SASS output to confirm whether the instruction exists in your binary.

---

## Suggested reproduction steps for a bug report

1. Provide the following outputs and attach them to the report:

   - `nvcc -V` (compiler version)
   - Output of `nvidia-smi` and `deviceQuery` (GPU model and compute capability)
   - The exact `nvcc` command-line used for both the debug and release builds (including all `-gencode`/`-arch` flags)
   - `cuobjdump --dump-sass ./a.out > sass.txt` (attach `sass.txt`)
   - `cuda-memcheck --tool=memcheck ./a.out` run output (attach logs)

2. Point to the lines/files where the `CudaDMAStrided`/`start_async_dma` implementation lives — in this project these are in `/mnt/data/jacobi2D_cudaDMA.cuh` and `/mnt/data/jacobi2D_cudaDMA.cu`.

---

## Example nvcc commands

- Debug (works, but slow):

```bash
nvcc -G -g -O0 /mnt/data/jacobi2D_cudaDMA.cu -o jacobi_dbg
```

- Release (may crash if not fixed):

```bash
nvcc -O3 /mnt/data/jacobi2D_cudaDMA.cu -o jacobi_release
```

- Multi-target (safe minimum + Ampere fast path):

```bash
nvcc -O3 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_80,code=sm_80 \
  /mnt/data/jacobi2D_cudaDMA.cu -o jacobi_multi
```

---



_End of README._
