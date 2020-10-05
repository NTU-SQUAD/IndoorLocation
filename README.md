# IndoorLocation

## Parameters
```
task: task number, Int, [1-3], default:1
number: show nth path of a floor, default:2
input: selet the data dir of which floor, default: `./data/site1/F1`
bssid: the wifi bssid for task 3, defualt: ''(will plot 3 default wifi heat map)
```

## Run

```
1. Task1
python3 main.py -t 1 -n 0

2. Task2
python3 main.py -t 2

3. Task3
python3 main.py -t 3
```