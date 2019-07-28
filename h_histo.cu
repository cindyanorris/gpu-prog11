#include <stdio.h>
#include "config.h"
#include "histogram.h"
#include "h_histo.h"
#include "CHECK.h"
#include "wrappers.h"

void histoOnCPU(histogramT *, unsigned char * Pin, int height, int width);
/*
   h_histo
   Builds the histogram of an image on the CPU.

   Phisto - pointer to a histogram struct that contains the bins
            to be filled
   Pin - array that contains the color pixels.
   width and height - dimensions of the image.

   Returns the amount of time it takes to build a histogram
*/
float h_histo(histogramT * Phisto, unsigned char * Pin,
             int height, int width)
{
    
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

    //build the histogram
    histoOnCPU(Phisto, Pin, height, width);

    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*
   histoOnCPU
   Builds the histogram of an image on the CPU.

   Phisto - pointer to a histogram struct that contains the bins
            to be filled
   Pin - array that contains the color pixels.
   width and height - dimensions of the image.
*/
void histoOnCPU(histogramT * Phisto, unsigned char * Pin, int height, int width)
{
    unsigned char redVal, greenVal, blueVal;
    int j, i; 

    //calculate the row width of the input 
    int rowWidth = CHANNELS * width;
    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i++)
        {
            redVal = Pin[j * rowWidth + i * CHANNELS]; 
            greenVal = Pin[j * rowWidth + i * CHANNELS + 1]; 
            blueVal = Pin[j * rowWidth + i * CHANNELS + 2]; 
            int bin = (redVal/TONESPB)*BINS*BINS + (blueVal/TONESPB)*BINS 
                      + greenVal/TONESPB;
            Phisto->histogram[bin]++;
        }
    }
}
