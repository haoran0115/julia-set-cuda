# Julia Set by CUDA
![](./julia.png)

A simple multi-thread CUDA program which generates the density map of [Julia set](https://en.wikipedia.org/wiki/Julia_set) for

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?f_C(Z)&space;=&space;Z^2&space;&plus;&space;C,&space;Z\in\mathbb{C},&space;C&space;=&space;e^{i\theta}">
</p>

To plot `.png` file, you should first compile the main CUDA program through `cmake`, please check `run.sh`. The library to plot `.png` is taken from [stb](https://github.com/nothings/stb).
```bash
./run.sh
```
You may modify some important parameters like iteration depth and xy range, please check `julia.cu`.

