#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
//config.h contains a number of configuration parameters
#include "config.h"
#include "histogram.h"
#include "d_histo.h"
#include "CHECK.h"

#define BLOCKDIM 32 //block dimension for both x and y

//prototype for the kernel
__global__ void d_histoKernel(histogramT *, unsigned char *, int, int, int);
__global__ void d_histoTileKernel(histogramT *, unsigned char *, int, int, int, int, int);
__global__ void d_histoSharedKernel(histogramT *, unsigned char *, int, int, int, int, int);
__global__ void d_histoPrivateKernel(histogramT *, unsigned char *, int, int, int, int, int);
__global__ void emptyKernel();


/*
   d_histo
   Builds a histogram of an image on the GPU.

   Phisto - pointer to a histogram struct that contains the bins
            to be filled
   Pin - array contains the color pixels to be histored.
   width and height -  dimensions of the image.
   tileHeight, tileWidth - width and height of the tile to be used
                           by the tiled versions
   which - indicates the version tu run
 
   Returns the amount of time it takes to perform the histo 
*/
float d_histo(histogramT * Phisto, unsigned char * Pin,
              int height, int width, int tileHeight, 
              int tileWidth, int which) 
{
    cudaEvent_t start_cpu, stop_cpu;
    int pitch;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    unsigned char * d_Pin;
    histogramT * d_Phisto;

    //create arrays in the GPU memory (DONE)
    CHECK(cudaMallocPitch((void **)&d_Pin, (size_t *) &pitch,
                          (size_t) (width * CHANNELS),
                          (size_t) height));
    for (int i = 0; i < height; i++)
       CHECK(cudaMemcpy(&d_Pin[i * pitch], &Pin[i * width * CHANNELS],
             width * CHANNELS, cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void **)&d_Phisto, sizeof(histogramT)));
    //initialize histogram to 0
    CHECK(cudaMemcpy(d_Phisto, Phisto, sizeof(histogramT), cudaMemcpyHostToDevice));

    //launch an empty kernel to get more accurate timing
    emptyKernel<<<1024, 1024>>>();

    CHECK(cudaEventRecord(start_cpu));
    if (which == NOTILE)
    {
        //define grid and block dimensions and launch the kernel
        dim3 grid(ceil(width/(float)BLOCKDIM), ceil(height/(float)BLOCKDIM), 1);
        dim3 block(BLOCKDIM, BLOCKDIM, 1);
        d_histoKernel<<<grid, block>>>(d_Phisto, d_Pin, height, width, pitch);  
    } else if (which == TILE)
    {
        //TO DO
        //define grid and block dimensions and launch the kernel
        //note that the number of blocks will depend upon the tile dimensions
        //d_histoTileKernel<<<grid, block>>>(d_Phisto, d_Pin, height, width, 
        //                                   pitch, tileHeight, tileWidth);
    } else if (which == SHARED)
    {
        //TO DO
        //define grid and block dimensions and launch the kernel
        //note that the number of blocks will depend upon the tile dimensions
        //d_histoSharedKernel<<<grid, block>>>(d_Phisto, d_Pin, height, width, 
        //                                     pitch, tileHeight, tileWidth);  
    } else if (which == PRIV)
    {
        //TO DO
        //define grid and block dimensions and launch the kernel
        //note that the number of blocks will depend upon the tile dimensions
        //d_histoPrivateKernel<<<grid, block>>>(d_Phisto, d_Pin, height, width, 
        //                                     pitch, tileHeight, tileWidth);  
    }

    //the rest of this function is COMPLETE
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(Phisto, d_Phisto, sizeof(histogramT), cudaMemcpyDeviceToHost));

    //record the ending time and wait for event to complete
    CHECK(cudaEventSynchronize(stop_cpu));
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*
   d_histoKernel
   Kernel code executed by each thread on its own data when the kernel is
   launched. A single thread handles one piece of data in the Pin
   array and increments one bin.

   histo - pointer to the histogram struct that is in global memory
   Pin - array contains the color pixels to be used for the histogram
   width and height -  dimensions of the image.
   pitch - size of each row (includes CHANNELS).
 
*/
__global__
void d_histoKernel(histogramT * histo, unsigned char * Pin, int height,
                  int width, int pitch)
{
    //THIS FUNCTION IS COMPLETE.

    //choose the single pixel to be handled by the thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width)
    {
        int pIndx = row * pitch + col * CHANNELS;
        //use the red, green and blue values to determine the bin
        unsigned char redVal = Pin[pIndx];
        unsigned char greenVal = Pin[pIndx + 1];
        unsigned char blueVal = Pin[pIndx + 2];
        int bin = (redVal/TONESPB)*BINS*BINS + (blueVal/TONESPB)*BINS
                      + greenVal/TONESPB;
        atomicAdd(&(histo->histogram[bin]), 1); 
    }
}

/*
   d_histoTileKernel
   Kernel code executed by each thread on its own data when the kernel is
   launched. A single thread handles a tile of input whose size
   is tileHeight by tileWidth.

   histo - pointer to the histogram struct that is in global memory
   Pin - array contains the color pixels to be used for the histogram
   width and height -  dimensions of the image.
   pitch - size of each row (includes CHANNELS).
   tileHeight and tileWidth - height and width of the tile of pixels
                              handled by one thread
*/
__global__
void d_histoTileKernel(histogramT * histo, unsigned char * Pin, int height,
                  int width, int pitch, int tileHeight, int tileWidth)
{
    //Visit all pixels in a tile and increment bins.
    //Don't put the tile in the shared memory.  There is no reason
    //to do this since each pixel is only visited once.  
    //This version is different from the last kernel in that each 
    //thread has more work to do; it operates on tileHeight * tileWidth pixels 
    //instead of just one.
}


/*
   d_histoSharedKernel
   Kernel code executed by each thread on its own data when the kernel is
   launched. A single thread handles a tile of input whose size
   is tileHeight by tileWidth. In addition, shared memory is used for
   a block level copy of the bins to reduce the number of writes to
   global memory.

   histo - pointer to the histogram struct that is in global memory
   Pin - array contains the color pixels to be used for the histogram
   width and height -  dimensions of the image.
   pitch - size of each row (includes CHANNELS).
   tileHeight and tileWidth - height and width of the tile of pixels
                              handled by one thread
*/
__global__
void d_histoSharedKernel(histogramT * histo, unsigned char * Pin, int height,
                  int width, int pitch, int tileHeight, int tileWidth)
{
    //bins that are shared by all threads in a block
    __shared__ unsigned int sBin[TOTALBINS];

    //use threads in the block cooperate in initializing the
    //shared bin

    //visit all pixels in the tile and increment shared bins

    //threads in a block cooperate in adding the shared bin values to
    //the global bins
}

/*
   d_histoPrivateKernel
   Kernel code executed by each thread on its own data when the kernel is
   launched. A single thread handles a tile of input whose size
   is tileHeight by tileWidth. In addition, shared memory is used for
   a block level copy of the bins to reduce the number of writes to
   global memory. Finally, this kernel also reduces the number of
   atomicAdds by using the privatization technique discussed on page 212.

   histo - pointer to the histogram struct that is in global memory
   Pin - array contains the color pixels to be used for the histogram
   width and height -  dimensions of the image.
   pitch - size of each row (includes CHANNELS).
   tileHeight and tileWidth - height and width of the tile of pixels
                              handled by one thread
*/
__global__
void d_histoPrivateKernel(histogramT * histo, unsigned char * Pin, int height,
                  int width, int pitch, int tileHeight, int tileWidth)
{
    //bins that are shared by all threads in a block
    __shared__ unsigned int sBin[TOTALBINS];

    //visit every pixel in a tile, updating sBin
    //use the privatization technique on page 212 to reduce
    //accessed to shared memory
    //However, there are two bugs in that code.
    //1) the atomicAdd should use prev_index, not alphabet_position/4 as the index
    //2) after loop, update sBin if accumulator is greater than 0

    //all threads cooperate in adding shared bins to global bins
}


__global__ void emptyKernel()
{
    //launched to get more accurate timing
}
