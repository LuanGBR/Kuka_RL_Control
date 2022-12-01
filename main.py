from src.Training import TrainingEnv
from src.Evaluation import EvaluationEnv
import numpy as np
from os import path
import json
import argparse







def main(args):
    #read params
    with open(args.paramfile) as f:
        params = json.load(f)
    
    params["cam_R"] = np.array(params["cam_R"])
    params["cam_t"] = np.array(params["cam_t"])
    

    if args.mode == "train":
        print("Training the model")
        training = TrainingEnv(params)
        training.train()
    elif args.mode == "eval":
        print("Evaluating the model")
        evaluation = EvaluationEnv(args.model,params)
        evaluation.evaluate()
    else:
        raise ValueError("Invalid mode")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #read arguments from command line
    parser.add_argument("mode",choices=["train","eval"],help="train or evaluate the model")
    #params filepath
    parser.add_argument('-pf',"--paramfile",default="params.json",help="path to params.json file")
    #eval model filepath
    parser.add_argument("-m","--model",default="./saved_data/target.pth",help="path to model.pth file in eval mode")



    args = parser.parse_args()
    main(args)
        
    

    






 