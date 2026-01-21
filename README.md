# dealii-step67-supersonic
This is an adaptation of tutorial step-67 of deal.II for supersonic flows.

Installation of deal.II version v9.3.0 or v9.5.1 is required.  
For more information on how to install, please see (https://dealii.org/current/readme.html).  
If you are using Mac OS X, please see (https://github.com/dealii/dealii/wiki/MacOSX).

You can control execution with the variables on the top of the source file.  

### To compile and run the code:  

1. Start a shell inside the folder.
2. Then type following commands
```
cmake .
make
mpirun -np 4 ./step-67
``` 

Results are in the ./results folder.<br/>
You can open the results in Paraview.
