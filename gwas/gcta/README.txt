PROGRAM: GCTA

DESCRIPTION: Genome-wide Complex Trait Analysis

AUTHOR: Jian Yang, Hong Lee, Mike Goddard, Zhili Zheng, Jian Zeng, Zhihong Zhu, 
Andrew Bakshi, Robert Marie and Peter Visscher

CONTACT: jian.yang@uq.edu.au

2010-2017 

The binary are released under MIT License. As the license of the dependency packages, 
you shall turn to the licenses for the usage other than academic.

DOCUMENTATION: http://cnsgenomics.com/software/gcta/

INSTALLATION: When you have downloaded a zip or gzipped archive with an
executable binary, no installation is necessary. NOTE: If there is a 
bin folder, you should put all files in that folder all together, copy
gcta64 out only will not work.

USAGE: Type "gcta64" or "./gcta64" from the command line followed by the
options of choice (see documentation) NOTE: you probably need to run 
"chmod a+x gcta" to get the correct permission to execute the program.

EXAMPLE DATA: Four example files test.bed, test.bim, test.fam and test.phen 
are included in the distribution; for example, once GCTA is installed try running:

     gcta64 --bfile test --make-grm --out test

     gcta64 --reml --grm test --pheno test.phen --out test

     etc...

In Mac or Windows version, the example files locates in the parent folder of gcta64. 

