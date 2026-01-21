# Hello! Welcome to dealii-step67-supersonic :)

Follow the instructions below to view, run, and edit the container.
  
  
## Get docker image from dockerhub:
Start docker and login  

docker image pull mariastf/dealii-step67-supersonic:latest  

if you face platform errors when pulling, maybe you are a mac user? try    
docker pull --platform=linux/amd64 mariastf/dealii-step67-supersonic:latest  
append --platform=linux/amd64 in every docker command for this image

## To inspect content, start a shell inside the image:
docker run -it mariastf/dealii-step67-supersonic bash  
ls  
exit

## Run and Get Results in a Volume mount:
go to a directory of choice  

docker run --rm -v $(pwd)/results:/home/dealii_user/step-67-supersonic/results mariastf/dealii-step67-supersonic  
open results in paraview

## Copy source code on a linux mount:
docker create --name extract mariastf/dealii-step67-supersonic  
docker cp extract:/home/dealii_user/step-67-supersonic .  
docker rm extract   
cd step-67-supersonic  

now you should be able to view the source code and edit it  
you can control execution with the variables on the top of the source file, edit and save

start shell inside the container:

docker run -it --rm -v $(pwd):/home/dealii_user/step-67-supersonic mariastf/dealii-step67-supersonic bash  
cmake .  
make  
mpirun -np 4 ./step-67    
open results in paraview

exit