#!/bin/bash



# Set variables
dir_to_run="./tdscha_raman_FT_0"
output="output_0.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_raman_FT_1"
output="output_1.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_raman_FT_2"
output="output_2.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_raman_FT_3"
output="output_3.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_raman_FT_4"
output="output_4.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_raman_FT_5"
output="output_5.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait


# Set variables
dir_to_run="./tdscha_raman_FT_6"
output="output_6.txt"

# Run the command
# COMMAND="nohup mpirun -np 8 python-jl ./run.py $dir_to_run > $output &"
COMMAND="nohup python-jl ./run.py $dir_to_run > $output &"
echo "Running command: $COMMAND"
$COMMAND

# Wait for the command to finish
wait
