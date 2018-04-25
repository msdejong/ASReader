python -m cProfile -o attention.cprof  main.py --train_path /home/ubuntu/drive/home/ubuntu/747/questions/training --valid_path /home/ubuntu/drive/home/ubuntu/747/questions/validation \
--test_path /home/ubuntu/drive/home/ubuntu/747/questions/test --eval_interval 500 --use_cuda True --num_epochs 2 --learning_rate 0.001 --seed 2 --batch_size 16


