echo "Removing any jpg files"
rm -f *.jpg

echo "Generating jpg files"
#define an array of image sizes 'width:height'
sizes=('5000:4000' '4000:5000' '5000:5000' '7500:5000')
for size in ${sizes[@]}; do
   if [[ $size =~ ^([0-9]+):([0-9]+)$ ]]; then
      width="${BASH_REMATCH[1]}";
      height="${BASH_REMATCH[2]}";
      filename="color${width}by${height}.jpg"
      ./generate $filename $width $height
   fi
done

#define an array of tiles sizes to test 'x:y'
blkSizes=('1:1' '2:2' '4:2' '4:2' '4:4' '8:2')

for size in ${sizes[@]}; do
   if [[ $size =~ ^([0-9]+):([0-9]+)$ ]]; then
      width="${BASH_REMATCH[1]}";
      height="${BASH_REMATCH[2]}";
      filename="color${width}by${height}.jpg"
      echo "---------------------";
      for bsize in ${blkSizes[@]}; do
         if [[ $bsize =~ ^([0-9]+):([0-9]+)$ ]]; then
            tw="${BASH_REMATCH[1]}";
            th="${BASH_REMATCH[2]}";
            echo " "
            echo "./histo -w $tw -h $th $filename"
            ./histo -w $tw -h $th $filename
         fi
      done
   fi
done

echo "Removing jpg files"
rm -f *.jpg

