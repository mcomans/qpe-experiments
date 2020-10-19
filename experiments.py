import os
import pandas as pd
import subprocess
import re

USER = "am72ghiassi"

experiments = pd.read_csv("experiments.csv")
pd.DataFrame(columns=experiments.columns)

results = []

with open("results.csv", "w") as f:
    f.write("exp,replication,learning_rate,batch_size,epochs,train_size,model,accuracy\n")

for index, experiment in experiments.iterrows():
    model = experiment["model"]

    options = ["--action", "train", 
            "--dataPath", "/tmp/mnist",  
            "--batchSize", f"{experiment['batch_size']}", 
            "--endTriggerNum", f"{experiment['epochs']}", 
            "--learningRate", f"{experiment['learning_rate']}", 
            "--train-size", f"{experiment['train_size']}"]
    options_string = ' '.join(options)

    if model == "lenet5":
        cmd = ' '.join([f"/home/{USER}/bd/spark/bin/spark-submit", 
            "--master", "spark://10.164.0.2:7077", 
            "--driver-cores", "2",
            "--driver-memory", "2G", 
            "--total-executor-cores", "4", 
            "--executor-cores", "2", 
            "--executor-memory", "2G",
            "--py-files", f"/home/{USER}/bd/spark/lib/bigdl-0.11.0-python-api.zip,codes/lenet5.py", 
            "--properties-file", f"/home/{USER}/bd/spark/conf/spark-bigdl.conf", 
            "--jars", f"/home/{USER}/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar", 
            "--conf", f"spark.driver.extraClassPath=/home/{USER}/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar",
            "--conf", f"spark.executer.extraClassPath=bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar codes/lenet5.py {options_string}", 
            ])
    if model == "bi-rnn":
        cmd = ' '.join([f"/home/{USER}/bd/spark/bin/spark-submit", 
            "--master", "spark://10.164.0.2:7077", 
            "--driver-cores", "2",
            "--driver-memory", "2G", 
            "--total-executor-cores", "4", 
            "--executor-cores", "2", 
            "--executor-memory", "2G",
            "--py-files", f"/home/{USER}/bd/spark/lib/bigdl-0.11.0-python-api.zip,codes/bi-rnn.py,codes/utils.py", 
            "--properties-file", f"/home/{USER}/bd/spark/conf/spark-bigdl.conf", 
            "--jars", f"/home/{USER}/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar", 
            "--conf", f"spark.driver.extraClassPath=/home/{USER}/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar",
            "--conf", f"spark.executer.extraClassPath=bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar codes/bi-rnn.py {options_string}", 
            ])

    for i in range(5):
        print(f"Running experiment {experiment['exp']}, replication {i+1}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)

        with open(f"exp{experiment['exp']}-rep{i+1}.log", "wb") as log_file:
            log_file.write(result.stdout)
        last_line = result.stdout.split(b'\n')[-2].decode('utf-8')
        pattern = '^Evaluated result: (\d\.\d+)'
        try:
            accuracy = re.search(pattern, last_line).group(1)
        except:
            print(f"Exception occurred in extracting accuracy, see log file exp{experiment['exp']}-rep{i+1}.log")
            accuracy = "NaN"
        with open("results.csv", "a") as f:
            f.write(f"{experiment['exp']},{i+1},{experiment['learning_rate']},{experiment['batch_size']},{experiment['epochs']},{experiment['train_size']},{experiment['model']},{accuracy}\n")
    

