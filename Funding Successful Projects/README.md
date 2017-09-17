## Machine Learning Challenge #2

This repository contains Python script and sample submission file. The code is build using spyder software. The name of solution code file is Kickstarter.py. The  samplesubmission.csv file contain output result obtained after running Kickstarter.py script. To download dataset for this problem click on Download dataset option at the end of this file. This problem statement and data set is taken from HackerEarth.com.

## Problem Statement:-Funding Successful Projects

Kickstarter is a community of more than 10 million people comprising of creative, tech enthusiasts who help in bringing creative project to life. Till now, more than $3 billion dollars have been contributed by the members in fuelling creative projects. The projects can be literally anything – a device, a game, an app, a film etc.

Kickstarter works on all or nothing basis i.e if a project doesn’t meet it goal, the project owner gets nothing. For example: if a projects’s goal is $500. Even if it gets funded till $499, the project won’t be a success.

Recently, kickstarter released its public data repository to allow researchers and enthusiasts like us to help them solve a problem. Will a project get fully funded ?

In this challenge, you have to predict if a project will get successfully funded or not.

## Data Description

There are three files given to download: train.csv, test.csv and sample_submission.csv The train data consists of sample projects from the May 2009 to May 2015. The test data consists of projects from June 2015 to March 2017. 

| Variable | Description |
| --- | --- |
| `project_id` | unique id of project |
| `name` | name of the project |
| `desc` | description of project |
| `goal` | the goal (amount) required for the project |
| `keywords` | keywords which describe project |
| `disable communication` | whether the project authors has disabled communication option with people donating to the project |
| `country` | country of project author |
| `currency` | currency in which goal (amount) is required |
| `state_changed_at` | at this time the project status changed. Status could be successful, failed, suspended, cancelled etc. (in unix timeformat) |
| `created_at` | at this time the project was posted on the website(in unix timeformat) |
| `launched_at` | at this time the project went live on the website(in unix timeformat) |
| `backers_count` | no. of people who backed the project |
| `final_status` | whether the project got successfully funded (target variable – 1,0) |

## Download Dataset
* [Download Dataset](https://he-s3.s3.amazonaws.com/media/hackathon/machine-learning-challenge-2/funding-successful-projects/3149def2-5-datafiles.zip)
