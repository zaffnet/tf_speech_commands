for i in  0.00001 0.0001 0.001 0.01 0.1 1.0
do
  python train.py --data_url= --data_dir=/home/zafar/speech_dataset/ --how_many_training_step=500 --batch_size=50 \
  --model_architecture=conv_batchnorm --learning_rate=$i --summaries_dir=/home/zafar/Desktop/logs4/adam$i/
done