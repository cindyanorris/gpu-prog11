#define CHANNELS 3   //red, green blue
#define TONES 256    //colors are values between 0 and 255
//number of bins in one dimension of the three dimensions; space is 3D
//red bins by green bins by blue bins
//Can change this to another factor of 256 to experiment with
//different bin sizes, but if it gets too large, there will be
//insufficient shared memory
#define BINS 8      
//tones per bin 
#define TONESPB (TONES/BINS)
//total number of bins as one dimension
#define TOTALBINS (BINS*BINS*BINS)
//filename length
#define NAMELEN 80
//which GPU algorithm to use
#define TILE 1
#define NOTILE 2
#define SHARED 3
#define PRIV 4
