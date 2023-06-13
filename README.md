Churn experimenting
==============================

This repository contains various attempts to forecast churn.

## 1. Setup

### 1.1 Setting up your Python environment
 
The steps below describe how to set up your environment for Churn experiment.
For this we assume you have a terminal open and have it pointed at the root of this project.

1. Install the conda environment: `conda env create -f environment.yml`
2. Activate the environment: `conda activate churn`
3. Install the source code as a Python module: `pip install -e .`
4. (Optional) Install Plotly widgets for Jupyter: `jupyter labextension install jupyterlab-plotly@4.9.0`  
Note that for this step you need node installed, either on your system itself, or through conda. 
   
### 1.2 Setting up your IDE
This part is set up under the assumption that PyCharm will be used.
1. Set up the Churn interpreter (`churn`) when opening this project in PyCharm. 
2. Set up Black. Black is a code formatter with quite strict opinions. While not all the choices it
 makes are optimal, it does make sure that the code everyone commits is exactly the same.
 [This](https://black.readthedocs.io/en/stable/editor_integration.html#pycharm-intellij-idea) shows
 how to set up Black in PyCharm. 
 Though there are three additional notes:
   1. The most important part of that setup is having it run on every save using the file watcher
   plugin.
   2. I'd suggest installing Black in a more global environment rather than `churn`. Reason for this
   is that the Black file watcher setup can be shared over all projects, and it's better not to
   have it linked to a specific environment then.
   3. We set the line length to 99 characters. So make sure that the argument is `-l 99 $FilePath$`
   instead of just `$FilePath$`.
3. Add a line length indicator to PyCharm to match the line length used by Black. This can be
configured at `Settings -> Editor -> Code Style -> Hard wrap at` and should be set to 99. This is
useful for things like comments and documentation, because Black doesn't change comments.

While the above steps use PyCharm as IDE, all these things can also be configured in other
IDEs/editors like Visual Studio Code. 

## 2. Running the code

### 2.1 Secrets

For this exploration of churn, a Kaggle dataset has been used. The data can be downloaded from Kaggle using the Kaggle-api. This api needs a username and key which are defined in the `.env` file. See `.env_example` on how to structure this file.

### 2.2 Attempts

The main scripts of the churn exploration are located in the `scr` directory:

* `churn_blog.py` contains the code to reproduce the rond blog on churn: https://medium.com/rond-blog/customer-churn-e94d5b6eee23
* `predict_churn_follwo_blog.py` contains the code that follow the explanation in here (amongst others): https://towardsdatascience.com/hands-on-predict-customer-churn-5c2a42806266. See also the script itself for further explanation.
* `survival_analysis_for_churn.py` initial exploration of the data.
* `` creation of a simple churn model to be used for deployment.

## 3. Deployment

This repository also contains several folders with different try-outs of api's. All of them make use of `FastApi` package.
Deployment is done using either Deta, or Azure. Deta only allows uploading relatively small packages (code and
libraries). However we need to deploy a survival model, provided by the `lifelines` package. This package is
too large for Deta, therefore Azure is used.

### 3.1 Local

The api can be tested locally by running `uvicorn main:app --reload` from the directory of the particular deployment (e.g.  the `purple_survival` folder).


### 3.2 Azure

Steps needed to deploy on Azure (assuming the web-app and container registry have been created already):

1. Move to the `purple_survival` folder.
2. Build a docker image using `docker build -t survival .`.
3. Start a docker container based on the survival docker image
   using `docker run -dp 8000:80 --env API_KEY=secret_api_key survival`. The secret key is provided through a
   environment variable.
4. Verify the container works properly by accessing http://localhost:8000/.
5. Retag the image by `docker tag survival purplecontainer.azurecr.io/survival`.
6. Log in bij `az login` met de eigen credentials via de geopende
webpagina
7. Also login at the container registry (if needed) using `az acr login --name purplecontainer`.
7. Push to Azure container registry by `docker push purplecontainer.azurecr.io/survival`.

The correct working can also be tested using the file `apply_survival_api.py`.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── deployments              <- Several FastApi deployments
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml   <- The used to (re)create the environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
