#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <jpeglib.h>    
#include <jerror.h>

void checkCommandArgs(int, char **, int *, int *);
void printUsage();
void createJPGImage(FILE *, int, int);
 
int main(int argc, char * argv[])
{
    int width, height;
    checkCommandArgs(argc, argv, &width, &height);
    FILE * fp = fopen(argv[1], "wb");
    if (fp == NULL) 
    {
        printf("\nFile open failed.\n\n");
        printUsage();
    }
    createJPGImage(fp, width, height);
    return EXIT_SUCCESS;
}

void  createJPGImage(FILE * fp, int width, int height)
{
   struct jpeg_compress_struct cinfo;
   struct jpeg_error_mgr jerr;
   int rowStride;
   JSAMPROW rowPointer[1];
   int channels = 3;
   J_COLOR_SPACE colorspace = JCS_RGB;
   unsigned char * line;
   int i, j;


   //set up error handling
   cinfo.err = jpeg_std_error(&jerr);
   //initialize the compression object
   jpeg_create_compress(&cinfo);

   //initalize state for output to outfile
   jpeg_stdio_dest(&cinfo, fp);

   cinfo.image_width = width;   
   cinfo.image_height = height;
   cinfo.input_components = channels;   
   cinfo.in_color_space = colorspace;
   jpeg_set_defaults(&cinfo);
   jpeg_set_quality(&cinfo, 75, TRUE);

   //TRUE means it will write a complete interchange-JPEG file
   jpeg_start_compress(&cinfo, TRUE);
   //bytes per row
   rowStride = width * channels;

   //while (cinfo.next_scanline < cinfo.image_height)
   //{
    //  rowPointer[0] = &image[cinfo.next_scanline * rowStride];
     // (void) jpeg_write_scanlines(&cinfo, rowPointer, 1);
  // }


   line = malloc(sizeof(unsigned char) * channels * width);
   rowPointer[0] = line;
   for (j = 0; j < height; ++j)
   {
      for (i = 0; i < width; ++i)
      {
         line[i * channels] = i % 256;  // red 
         line[i * channels + 1] = j % 256;  // green 
         line[i * channels + 2] = (i * j) % 256;  // blue 
      }
      (void) jpeg_write_scanlines(&cinfo, rowPointer, 1);
   }
   jpeg_finish_compress(&cinfo);
   fclose(fp);
   jpeg_destroy_compress(&cinfo);
}

void checkCommandArgs(int argc, char * argv[], int * width, int * height)
{
    if (argc != 4)
    {
        printUsage();
    }
    int len = strlen(argv[1]);
    if (len < 5) printUsage();
    if (strncmp(".jpg", &argv[1][len - 4], 4) != 0) printUsage();
    (*width) = atoi(argv[2]);
    (*height) = atoi(argv[3]);
    if ((*width) < 100 || (*height) < 100) printUsage();
}

void printUsage()
{
    printf("\nThis program creates a jpg file of the size indicated\n");
    printf("by the command line arguments.\n\n");
    printf("usage: generate <name>.jpg <width> <height>\n");
    printf("       <name>.jpg is the name of the created jpg file\n");
    printf("       width is the width of the jpg image in pixels\n");
    printf("       height is the height of the jpg image in pixels\n");
    printf("       both width and height must be >= 100\n\n");
    exit(EXIT_FAILURE);
}
