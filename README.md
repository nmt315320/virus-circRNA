# virus-circRNA
online web server is http://server.malab.cn/viral-CircRNA/.
dataset
Easy-Classify.py
 Nearest Neighbors

* GradientBoosting
* LibSVM
* Decision Tree
* Random Forest
* Naive Bayes
* ......
# Requirement
* python 3.7.4+numpy、scipy、pandas
# input
*A libsvm or arff format file containing all positive and negative examples
#output
results.xls
#command
python easy_classify.py -i train.libsvm -c 10
# Optional parameters:
 * `-o`: Specify the output excel file name. The default is results.xls. (If the result shows the following error: "Failed. Please close excel result file first.", the user must customize the output file name)
 * `-s`: Whether to find the best classifier parameters. `0` means not to seek, `1` means to seek. The default is `0`.
 * `-m`: Whether to operate in parallel, it is not recommended for large data sets above 1GB. `0` is a single-threaded operation, `1` is a multi-threaded parallel operation, the number of threads is the number of classifiers running at the same time, suitable for users with powerful CPU and memory resources. The default is `1`.

# Help:
```ssh
  python easy_classify.py -h
```
miRanda: miRanda<miRNA><mRNA>
 
TargetScan: Perl TargetScan.pl<miRNA><mRNA><TargetScan_out>; perl TargetScan_60_context_scores.pl<miRNA><mRNA><TargetScan_out><TargetScan_context_score_out>
 
CCmiR.py –a<miRNA>-b<mRNA>-c<miRNA expression level>-o<output>-P<number of used CPUs>  ---miRNA expression level is opitional. CCmiR url is http://hulab.ucf.edu/research/projects/miRNA/CCmiR/download/readme.txt
