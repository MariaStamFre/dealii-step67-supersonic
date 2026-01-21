# dealii-step67-supersonic
This is an adaptation of tutorial step-67 of deal.II for supersonic flows.

Installation of deal.II version v9.3.0 or v9.5.0 is required.  
For more information on how to install see (https://dealii.org/current/readme.html)  

You can control execution with the variables on the top of the source file.  
To compile and run:  

Start a shell inside in the folder
cmake .
make
mpirun -np 4 ./step-67 

results are in the ./results folder
you can open results in paraview