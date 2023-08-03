---
title: 'FrESCO: Framework for Exploring Scalable Computational Oncology'
tags:
   - Python
   - bioinformatics
   - natural language processing
   - deep learning
   - computational oncology
authors:
  - name: Adam Spannaus
    affiliation: 1
    corresponding: True
  - name: John Gounley
    affiliation: 1
  - name: Mayanka Chandra Shekar
    affiliation: 1
  - name: Zachary R. Fox
    affiliation: 1
  - name: Jamaludin Mohd-Yusof
    affiliation: 2
  - name: Noah Schaefferkoetter
    affiliation: 1
  - name: Heidi A. Hanson
    affiliation: 1
affiliations:
  - name: Oak Ridge National Laboratory, Oak Ridge, TN, USA
    index: 1
  - name: Los Alamos National Laboratory, Los Alamos, NM, USA
    index: 2
date: 8 March 2023
bibliography: shared/refs.bib
---

# Statement of Need

The National Cancer Institute (NCI) monitors population level cancer
trends as part of its Surveillance, Epidemiology, and End Results (SEER)
program. This program consists of state or regional level cancer
registries which collect, analyze, and annotate cancer pathology
reports. From these annotated pathology reports, each individual
registry aggregates cancer phenotype information from electronic health
records. This data is then used to create summary statistics about
cancer incidence and mortality to facilitate population health
monitoring. Extracting phenotypic information from these reports is a
labor intensive task, requiring specialized knowledge about the reports
and cancer. Automating the information extraction process from cancer
pathology reports has the potential to improve data quality by
extracting information in a consistent manner across registries. It can
also improve patient outcomes by reducing the time from diagnosis to
population health statistic and enabling rapid case ascertainment for
clinical trials. Here we present `FrESCO`, a modular deep-learning
natural language processing (NLP) library initially designed for extracting pathology
information from clinical text documents. Our library is not limited to 
clinical medical text, but may also be used by  
researchers just getting started with NLP methods and those
looking for a robust solution for their classification problems.

# State of the Field

Other software to meet the demanding challenges of bringing ML to
biomedical studies have emerged in recent years.
Monai [@cardoso2022monai] is oriented towards ML on medical imaging data
and FuseMedML [@golts2023fusemedml] creates general and multimodal data
structures that are useful for biomedical ML. Most similar to `FrESCO`
is PyHealth [@zhao2021pyhealth] though it is more broadly scoped,
focusing on MIMIC (Medical Information Mart for Intensive Care),
electronic intensive care unit (eICU), and
observational medical outcomes partnership common data model (OMOP-CDM)
databases. Biomedical libraries such as Med7 [@kormilitzin2021med7] and
EHRkit [@li2022ehrkit] focus on electronic health records in general and
machine learning tasks such as named-entity recognition and document
summarization. Our `FrESCO` library is singularly focused on cancer
pathology reports and provides the model building workflow for
auto-coding SEER pathology reports, which is a fundamental requirement
in a clinical deployment environment [@harris2022clinical].

# Summary

The FrESCO codebase provides a deep-learning Python package based on
PyTorch [@pytorch] for extracting information from clinical text. While
the software is designed for clinical tasks, it may also be used for
typical NLP tasks such as sentiment classification. Our flexible and
modular codebase provides independent modules for: (1) loading text data
and creating data structures, (2) building and training deep-learning
models, and (3) scoring and evaluating trained models. Provided within
the code repository are three model architectures to classify text data:

1.  the multi-task convolutional neural network
    of [@alawad2020automatic],

2.  the hierarchical self-attention network (HiSAN) described
    in [@GAO2019101726], and

3.  the case-level context model (CLC) of [@case-level-context], for
    hierarchical datasets.

Each of these models is available with the deep-abstaining classifier
(DAC) of [@dac], which is presently only available as part of the CANDLE
code repository [@Candle]. The DAC adds an additional "abstention" class
to the specified model so that the classifier may choose none of the
available labels for a given task. While each model may work on generic
data, the HiSAN and CLC architectures are specifically designed to work
with patient data and are not available in other software packages like
PyHealth [@zhao2021pyhealth]. As an example, the CLC model uses multiple
pathology reports linked to an individual patient in a hierarchical way.
We have adapted the FrESCO codebase from our workflow within an airgapped system
which uses patient health data that is not publicly available.
This is the same tool we use internally, aside from internal consistency checks,
we are making it publicly available
to work with user supplied text data, the only requirement being the
format of the data files, which is specified in the `README`.

We have intentionally written this library with a working knowledge of
Python as the only prerequisite. Those who are just getting started or
are experienced NLP researchers or practitioners will find the code easy
to understand and expand upon. For example, one may create a
state-of-the-art NLP model by simply editing the configuration file,
without touching a line of code. Lastly, as the model definitions are
independent modules, one may experiment with their own custom model
definitions within the training and evaluation framework developed
herein.

# Acknowledgements

This manuscript has been authored by UT-Battelle, LLC, under
contract DE-AC05-00OR22725 with the US Department of Energy (DOE).
The US government retains and the publisher, by accepting the
article for publication, acknowledges that the US government retains
a nonexclusive, paid-up, irrevocable, worldwide license to publish
or reproduce the published form of this manuscript, or allow others
to do so, for US government purposes. DOE will provide public access
to these results of federally sponsored research in accordance with
the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

# References

