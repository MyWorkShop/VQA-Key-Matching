# Documents

## Reading datasets

The reading process has two parts.

The first part is to use`tr_write.py` to decode the images, resize them to 256x256x3(as configured in `resize_size`) and then write them into a `.tfrecord` file in pair with its `imgid`. With writting it into a `.tfrecord` file, we can then read it much more easily and effectively.

When it comes to the second parts, we will use`tr_read.py`to read the data in the runtime. We will first ininlize the offical json processing api, to read the data in `Questions/ `and `Annotatons/`folders. Because the api will use too much memory, so we will then copy the data which we may use out and then delete the api. After that, the question will be processed. We will build a dictionary of the top 2048(as configured in `fixed_num`) most used words, other words will be ignored as empty word. The string of question will be transfer to an array of its words id in the dictionary and 