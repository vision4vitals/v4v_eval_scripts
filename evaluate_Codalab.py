import warnings
warnings.filterwarnings("ignore")

import sys
import os
import os.path
import numpy as np
import random
import csv
from glob import glob

#set random seed 
random.seed(0)


print("V4V evaluation script v1.0.0\n")
msg = 'Please create a `results.txt` file and zip it into `submission.zip` as described in the Codalab platform. Upload `submission.zip.`'


#################################################
### The evaluation script contains plenty of 
### file-format validation and verbose 
### error messages. The metrics that are used 
### for evaluation are present in the bottom 
### half of the code. Thanks.
### 
### For any queries/issues, please reach out to
### arevanur<at>andrew<dot>cmu<dot>edu
################################################


def eprint(message):
    prefix = '''
---------------------------------------------
Sorry, your submission could not be processed. 
Please check the error message below.
---------------------------------------------

'''
    sys.exit(prefix+message)



def validate_file_contents(csv_iter):
    contents = []
    for i, gtline in enumerate(csv_iter):
        gtline = [value.strip() for value in gtline]

        ## Sanity Check 1: Ensure that first cols are valid strings and rest is valid float w/o any nan
        vidname = gtline[0]
        if type(vidname) != str:
            eprint('On line number {}. Video name not recognized. {}'.format(i+1, msg))

        vidtype = gtline[1]
        if vidtype not in ['HR', 'RR']:
            eprint('On line number {}. Unrecognized type (HR or RR). {}'.format(i+1, msg))

        try:
            vidvalues = np.array(gtline[2:]).astype(np.float32)
        except:
            eprint('On line number {}. There is a unrecognized element in the predictions. The predictions should include only real numbers and should not include blank inputs or NaNs. Please inspect and correct the file. {}'.format(i+1, msg))

        if np.isnan(np.sum(vidvalues)):
            eprint('On line number {}. There is a NaN value in the list. Please remove NaNs from the file. {}'.format(i+1, msg))

        contents.append((vidname, vidtype, vidvalues))

    return contents


##########################################################
# Main: load and evaluate scripts
##########################################################

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
refsubdir = glob(os.path.join(input_dir, 'ref', '**'))[0]

if not os.path.isdir(submit_dir):
    print(submit_dir + " doesn't exist!")
    eprint(submit_dir + " doesn't exist! " + msg)
else:
    gt_f = open(os.path.join(refsubdir, "gt.txt"))
    try:
        results_f = open(os.path.join(submit_dir, "results.txt"))
    except IOError:
        eprint('Could not find `results.txt`. You might need to check your filename. ' + msg)
    
    gt_reader_iter = csv.reader(gt_f, delimiter=',')
    res_iter = csv.reader(results_f, delimiter=',')

    gt_contents = validate_file_contents(gt_reader_iter)
    result_contents = validate_file_contents(res_iter)

    if len(gt_contents) != len(result_contents):
        LEN_GT_CONTENTS = len(gt_contents)
        eprint('There should be {} rows in your submission. {}'.format(LEN_GT_CONTENTS, msg))


    hr_gt, rr_gt, hr_pred, rr_pred = [], [], [], []
    for i, (gtline, resline) in enumerate(zip(gt_contents, result_contents)):
        gtvidname, gtvidtype, gtvidvalues = gtline
        resvidname, resvidtype, resvidvalues = resline

        if gtvidname != resvidname:
            eprint('On line number {}. There is video name mismatch. {}'.format(i+1, msg))
            
        if gtvidtype == 'RR':
            if resvidtype != 'RR':
                eprint('On line number {}. An HR type was found when a RR prediction array was expected. Please indicate RR values as -1.0 for each frame if not participating the RR Subchallenge. {}'.format(i+1, msg))

        if gtvidtype != resvidtype:
            eprint('On line number {}. The type (HR or RR) has a mismatch. {}'.format(i+1, msg))


        LENGTH_gt = len(gtvidvalues)
        if LENGTH_gt != len(resvidvalues):
            eprint('On line number {}. There should be exactly {} predictions on this line. {}'.format(i+1, LENGTH_gt, msg))

        if gtvidtype == 'HR' and resvidtype == 'HR':
            hr_gt.append(gtvidvalues)
            hr_pred.append(resvidvalues)
        elif gtvidtype == 'RR' and resvidtype == 'RR':
            rr_gt.append(gtvidvalues)
            rr_pred.append(resvidvalues)
        else:
            eprint('On line number {}. Type mismatch on line number. {}'.format(i+1, msg))
        
    
    hr_gt = np.concatenate(hr_gt)
    rr_gt = np.concatenate(rr_gt)
    hr_pred = np.concatenate(hr_pred)
    rr_pred = np.concatenate(rr_pred)


    ### Check 4. check if RR is -1 and print if they are not participating in RR
    participating_in_rr = True
    if np.allclose(rr_pred, -1):
        print('User not participating in RR challenge.')
        participating_in_rr = False

    from sklearn.metrics import mean_absolute_error as compute_mae
    from sklearn.metrics import mean_squared_error as __mse
    compute_rmse = lambda x1, x2: np.sqrt(__mse(x1, x2))

    from scipy.stats import pearsonr
    compute_r = lambda x1, x2: pearsonr(x1, x2)[0]
    

    #######################################################################
    #### METRICS
    #######################################################################

    hr_mae = compute_mae(hr_pred, hr_gt)
    hr_rmse = compute_rmse(hr_pred, hr_gt)
    hr_r = compute_r(hr_pred, hr_gt)

    #write the result scores on all the videos
    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')
    output_file.write("HR-MAE:"+str(hr_mae)+'\n')
    output_file.write("HR-RMSE:"+str(hr_rmse)+'\n')
    output_file.write("HR-R:"+str(hr_r)+'\n')
    
    if participating_in_rr:
        rr_mae = compute_mae(rr_pred, rr_gt )
        rr_rmse = compute_rmse(rr_pred, rr_gt)
        rr_r = compute_r(rr_pred, rr_gt)
        output_file.write("RR-MAE:"+str(rr_mae)+'\n')
        output_file.write("RR-RMSE:"+str(rr_rmse)+'\n')
        output_file.write("RR-R:"+str(rr_r)+'\n')
    
    
    output_file.close()


print("done!")




