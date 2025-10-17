# GEMM using FP64 Cores

## To-Do

- [x] Implement basic kernel
- [x] Setup directory structure & Makefile
- [x] Add CUDA Error Handling
- [ ] Add 2-D Tiling
- [ ] Global Memory Access Optimizations
  - [ ] Batch Global Memory Accesses (Loop Unrolling)
    - [ ] Verify PTX & SASS
  - [ ] Chunk Glocal Memory Accesses (Force compiler to use widest load instructions)
    - [ ] Verify PTX & SASS
  - [ ] Overlap Global Memory Reads with Computations
- [ ] Shared Memory Access Optimizations
  - [ ] Add Register Tiling
  - [ ] Resolve Bank Conflicts
- [ ] Add Fine-Tuning Script
  - [ ] Fine-tune for A100's
- [ ] Add Double Buffering
- [ ] Add the Epilogue
- [ ] Basic Warp Specialization (DMA warps load next tile while compute warps process the current tile)
- [ ] ...More to come...

