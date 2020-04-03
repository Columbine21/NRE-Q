# Evaluate our model 

In this folder, we aim to evaluate our model result with the official evaluate tools.

This folder contains : **semeval2010_task8_scorer-v1.2.pl** , **semeval2010_task8_format_checker.pl**, **run_test.sh**, **test_keys.txt** files.

-   test_keys.txt : ground truth of test set 
-   run_test.sh : script used for evaluating automatically 
-   result.txt : the structural output file which contain the model prediction.
-   semeval2010_task8_scorer-v1.2.pl : the official scoring tool.
-   semeval2010_task8_format_checker.pl : the official format check tools.

## Usage

When you get your final result of the label of testset save in result.txt.

Your should first run this.

```sh
perl semeval2010_task8_format_checker.pl result.txt
```

If it shows format correct. Then use the run_test.sh and view your result in **result_scores.txt**.

```sh
./run_test.sh 
```

