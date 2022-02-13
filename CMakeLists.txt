project(julia-set-cuda)
enable_language(CUDA)
add_executable(julia julia.cu)
