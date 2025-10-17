# GEMM using FP64 Cores

## To-Do

- [x] Implement basic kernel
- [x] Setup directory structure & Makefile
- [x] Add CUDA Error Handling
- [x] Add 2-D Tiling
- [x ] Global Memory Access Optimizations
  - [x] Batch Global Memory Accesses (Loop Unrolling)
    - [ ] Verify PTX & SASS
  - [x] Chunk Glocal Memory Accesses (Force compiler to use widest load instructions)
    - [ ] Verify PTX & SASS
- [ ] Shared Memory Access Optimizations
  - [ ] Add Register Tiling
  - [ ] Resolve Bank Conflicts
- [ ] Overlap Global Memory Reads with Computations
- [ ] Add Fine-Tuning Script
  - [ ] Fine-tune for A100's
- [ ] Add Double Buffering
- [ ] Add the Epilogue
- [ ] Basic Warp Specialization (DMA warps load next tile while compute warps process the current tile)
- [ ] ...More to come...

