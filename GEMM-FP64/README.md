# GEMM using FP64 Cores

## To-Do

- [x] Implement basic kernel
- [x] Setup directory structure & Makefile
- [x] Add CUDA Error Handling
- [x] Add 2-D Tiling
- [ ] Global Memory Access Optimizations
  - [x] Batch Global Memory Accesses (Loop Unrolling)
    - [ ] Verify PTX & SASS
  - [x] Chunk Glocal Memory Accesses (Force compiler to use widest load instructions)
    - [ ] Verify PTX & SASS
- [ ] Shared Memory Access Optimizations
  - [x] Add Register Tiling
    - [ ] Verify PTX & SASS
  - [x] Resolve Bank Conflicts
- [x] Add the Epilogue
- [ ] Overlap Global Memory Reads with Computations
- [ ] Add Fine-Tuning Script
- [ ] Basic Warp Specialization (DMA warps load next tile while compute warps process the current tile)
- [ ] ...More to come...
