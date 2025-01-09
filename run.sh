env=CartPole-v1

python3 foundIT.py -env $env 
python3 utils/helpers.py --env $env --num 1 --cur 5-3 --samp 5 --iter 3
python3 foundIT.py -env $env 
python3 utils/helpers.py --env $env --num 2 --cur 5-3 --samp 5 --iter 3
python3 foundIT.py -env $env 
python3 utils/helpers.py --env $env --num 3 --cur 5-3 --samp 5 --iter 3

python3 foundIT.py -env $env -c
python3 utils/helpers.py --env $env --num c1 --cur 5-3 --samp 5 --iter 3
python3 foundIT.py -env $env -c
python3 utils/helpers.py --env $env --num c2 --cur 5-3 --samp 5 --iter 3
python3 foundIT.py -env $env -c
python3 utils/helpers.py --env $env --num c3 --cur 5-3 --samp 5 --iter 3