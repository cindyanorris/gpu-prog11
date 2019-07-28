#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>
//defines a number of configuration values used for building the histogram
#include "config.h" 
#include "histogram.h"
#include "wrappers.h"
#include "h_histo.h"
#include "d_histo.h"

//prototypes for functions in this file 
void parseCommandArgs(int, char **, char **, int *, int *);
void printUsage();
void readJPGImage(char *, unsigned char **, int *, int *);
void compare(histogramT *, histogramT *, const char *, int);
void initHistogram(histogramT *, char *, int);
void printHistogram(histogramT *);
void printBin(int *, int);

/*
    main 
    Opens the jpg file and reads the contents.  Uses the CPU
    and the GPU to build a histogram of the image.  Compares the CPU and GPU
    results.  Outputs the time of each.
*/
int main(int argc, char * argv[])
{
    unsigned char * Pin; 
    histogramT * h_hgram, * d_hgram;
    char * fileName;
    int width, height, tileHeight, tileWidth;
    float cpuTime, gpuTime;

    parseCommandArgs(argc, argv, &fileName, &tileHeight, &tileWidth);

    //create histogram structs for the host and the device
    h_hgram = (histogramT *) Malloc(sizeof(histogramT));
    d_hgram = (histogramT *) Malloc(sizeof(histogramT));
    initHistogram(h_hgram, fileName, TOTALBINS);

    printf("\nComputing histogram of %s. Tile width: %d. Tile height: %d.\n", 
            fileName, tileWidth, tileHeight);

    //read the image 
    readJPGImage(fileName, &Pin, &width, &height);

    //use the CPU to build the histogram
    cpuTime = h_histo(h_hgram, Pin, height, width);
    printf("\tCPU time: \t\t%f msec\n", cpuTime);

    //run a no-tiled version
    initHistogram(d_hgram, fileName, TOTALBINS);
    gpuTime = d_histo(d_hgram, Pin, height, width, 1, 1, NOTILE);
    compare(d_hgram, h_hgram, "untiled", TOTALBINS);
    printf("\tGPU (no tiled) time: \t%f msec\n", gpuTime);
    printf("\t            Speedup: \t%f\n", cpuTime/gpuTime);

    //run a tiled version
    initHistogram(d_hgram, fileName, TOTALBINS);
    gpuTime = d_histo(d_hgram, Pin, height, width, tileHeight, 
                      tileWidth, TILE);
    compare(d_hgram, h_hgram, "tiled", TOTALBINS);
    printf("\tGPU (tiled)    time: \t%f msec\n", gpuTime);
    printf("\t            Speedup: \t%f\n", cpuTime/gpuTime);

    //run a shared memory version
    initHistogram(d_hgram, fileName, TOTALBINS);
    gpuTime = d_histo(d_hgram, Pin, height, width, 
                      tileHeight, tileWidth, SHARED);
    compare(d_hgram, h_hgram, "shared", TOTALBINS);
    printf("\tGPU (shared)   time: \t%f msec\n", gpuTime);
    printf("\t            Speedup: \t%f\n", cpuTime/gpuTime);

    //run a privatization version
    initHistogram(d_hgram, fileName, TOTALBINS);
    gpuTime = d_histo(d_hgram, Pin, height, width, 
                      tileHeight, tileWidth, PRIV);
    compare(d_hgram, h_hgram, "private", TOTALBINS);
    printf("\tGPU (private)  time: \t%f msec\n", gpuTime);
    printf("\t            Speedup: \t%f\n", cpuTime/gpuTime);

    free(d_hgram);
    free(h_hgram);
    free(Pin);
    return EXIT_SUCCESS;
}

/* initHistogram
   This function takes as input an empty histogramT and initializes the
   bins to 0.  It also sets the fileName field of the histogramT struct
   to the name of the file containing the image that will be used to
   build the histogram.
*/ 
void initHistogram(histogramT * histP, char * fileName, int length)
{
    int i;
    strncpy(histP->fileName, fileName, NAMELEN);
    for (i = 0; i < length; i++)
    {
       histP->histogram[i] = 0;
    }
}

/* printHistogram
   This function takes as input a pointer to a histogram
   and prints out the bin values.
*/
void printHistogram(histogramT * histP, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        if (i % 16 == 0) printf("\n%3d: ", i);
        printf("%3d ", histP->histogram[i]);
    }
    printf("\n");
}

/* 
    compare
    This function takes two histograms and compares them. One of the histograms
    contains values calculated  by the GPU.  The other array contains
    values calculated by the CPU.  This function examines
    corresponding bins in the histograms to see that they match.

    dgram - histogram calculated by GPU
    hgram - histogram calculated by CPU
    length - number of bins
    version - width of image
    
    Outputs an error message and exits program if the arrays differ.
*/
void compare(histogramT * dgram, histogramT * hgram, const char * version, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        if (dgram->histogram[i] != hgram->histogram[i])
        {
            printf("%s histograms don't match\n", version);
            printf("host bin[%d] = %d\n", i, hgram->histogram[i]);
            printf("device bin[%d] = %d\n", i, dgram->histogram[i]);
            exit(EXIT_FAILURE);
        }
    }
}
   
/*
    readJPGImage
    This function opens a jpg file and reads the contents.  
    Each pixel consists of bytes for red, green, and blue.  
    The array Pin is initialized to the pixel bytes.  width and height
    are pointers to ints that are set to those values.
    fileName - name of the .jpg file
*/
void readJPGImage(char * fileName, unsigned char ** Pin, 
                  int * width, int * height)
{
  unsigned long x, y;
  unsigned long dataSize;             // length of the file
  int channels;                       //  3 =>RGB   4 =>RGBA 
  unsigned char * rowptr[1];          // pointer to an array
  unsigned char * jdata;              // data for the image
  struct jpeg_decompress_struct info; //for our jpeg info
  struct jpeg_error_mgr err;          //the error handler

  FILE* file = fopen(fileName, "rb"); //open the file

  info.err = jpeg_std_error(& err);
  jpeg_create_decompress(& info);     //fills info structure

  //if the jpeg file doesn't load
  if(!file) {
     fprintf(stderr, "Error reading JPEG file %s.\n", fileName);
     printUsage();
  }

  jpeg_stdio_src(&info, file);
  jpeg_read_header(&info, TRUE);   // read jpeg file header
  jpeg_start_decompress(&info);    // decompress the file

  //set width and height
  (*width) = x = info.output_width;
  (*height) = y = info.output_height;
  channels = info.num_components;
  if (channels != 3)
  {
     fprintf(stderr, "%s is not an RGB jpeg image\n", fileName);
     printUsage();
  }

  dataSize = x * y * channels;

  //--------------------------------------------
  // read scanlines one at a time & put bytes 
  //    in jdata[] array. Assumes an RGB image
  //--------------------------------------------
  jdata = (unsigned char *)Malloc(dataSize);
  while (info.output_scanline < info.output_height) // loop
  {
    // Enable jpeg_read_scanlines() to fill our jdata array
    rowptr[0] = (unsigned char *)jdata +  // secret to method
            channels * info.output_width * info.output_scanline;

    jpeg_read_scanlines(&info, rowptr, 1);
  }
  //---------------------------------------------------

  jpeg_finish_decompress(&info);   //finish decompressing
  jpeg_destroy_decompress(&info);

  fclose(file);                    //close the file
  (*Pin) = jdata;
  return;
}

/*
    parseCommandArgs
    This function parses the command line arguments. The program can be executed 
    like this:
    ./histo <file>.jpg
    or
    ./histo [-w <n>] [-h <n>] <file>.jpg
    The -w and -h options are used to specify the size of the tile used on
    the GPU for building the histogram.
    In addition, it checks to see if the last command line argument
    is a jpg file and sets (*fileNm) to argv[i] where argv[i] is the name of the ppm
    file.  
*/
void parseCommandArgs(int argc, char * argv[], char ** fileNm, 
                      int * tileHeight, int * tileWidth)
{
    int fileIdx = argc - 1, th = 1, tw = 1;
    struct stat buffer;

    for (int i = 1; i < argc - 1; i++)
    {
        
        if (strncmp("-h", argv[i], 3) == 0) 
        {
            if (i + 1 < argc) th = atoi(argv[i+1]);
            if (th == 0) th = 1;
            i++;
        } else if (strncmp("-w", argv[i], 3) == 0) 
        {
            if (i + 1 < argc) tw = atoi(argv[i+1]);
            if (tw == 0) tw = 1;
            i++;
        } else
            printUsage();
    } 

    //check the input file name (must end with .jpg)
    int len = strlen(argv[fileIdx]);
    if (len < 5) printUsage();
    if (strncmp(".jpg", &argv[fileIdx][len - 4], 4) != 0) printUsage();

    //stat function returns 1 if file does not exist
    if (stat(argv[fileIdx], &buffer)) printUsage();
    (*fileNm) = argv[fileIdx];
    (*tileHeight) = th;
    (*tileWidth) = tw;
}

/*
    printUsage
    This function is called if there is an error in the command line
    arguments or if the .jpg file that is provided by the command line
    argument is improperly formatted.  It prints usage information and
    exits.
*/
void printUsage()
{
    printf("This application takes as input the name of a .jpg\n");
    printf("file containing a color image and creates a file\n");
    printf("containing a histogram representation of the image.\n");
    printf("\nusage: histo [-h <n>] [-w <n>] <name>.jpg\n");
    printf("         -h <n> option can be used to provide the\n");
    printf("                height of the tile used by the GPU.\n");
    printf("         -w <n> option can be used to provide the\n");
    printf("                width of the tile used by the GPU.\n");
    printf("         Defaults for tile width and height are 1\n");
    printf("         <name>.jpg is the name of the input ppm file.\n");
    printf("Examples:\n");
    printf("./histo color1200by800.jpg\n");
    printf("./histo -h 3 -w 4 -save color1200by800.jpg\n");
    exit(EXIT_FAILURE);
}
