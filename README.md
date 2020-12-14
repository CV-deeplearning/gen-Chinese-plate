# gen-Chinese-plate
做车牌识别，缺乏数据时，可造车牌，提高识别模型的鲁棒性(造车牌数据的环境为python2)。

## 造普通车牌

-  运行以下命令，你会在./data/common_plate目录下生成100张普通车牌。
-  $ python gen_common_plate.py --make_num 100 --out_dir ./data/common_plate
## 造新能源车牌
-  运行以下命令，你会在./data/green_plate目录下生成100张新能源车牌。
-  $ python gen_green_plate.py --make_num 100 --out_dir ./data/green_plate

## Author
  + Guo Pei
  + 20191010
