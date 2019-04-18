##################################################################
#
#  Environment Initial Setup
#
##################################################################

VIRTUAL_ENV_NAME=cloud_ml

##################################################################
#
# Create Conda Virtual Env 
# NOTE: This only needs to be ran one time to get the venv setup
#
##################################################################

echo "[ INFO ] Creating conda virtual env ($VIRTUAL_ENV_NAME)"
conda create --name $VIRTUAL_ENV_NAME python=3.5 \
    numpy=1.14.5 \
    pandas=0.23.3 \
    scipy=1.1.0 \
    scikit-learn=0.19.2 \
    tabulate=0.8.2


# Remove Conda ENV
# conda remove --name $VIRTUAL_ENV_NAME --all


# List Conda ENVs
echo "[ INFO ] Listing Conda Virtual Environments..."
conda env list


#ZEND
