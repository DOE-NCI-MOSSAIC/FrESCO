import json
import os
import pandas as pd
from bardi.data import data_handlers
from bardi.pipeline import Pipeline
from bardi import nlp_engineering
from bardi.nlp_engineering.splitter import NewSplit
from bardi.nlp_engineering.regex_library.pathology_report import PathologyReportRegexSet


# create some sample data
# df = pd.DataFrame([
#     {
#         "patient_id_number": 1,
#         "text": "The patient presented with notable changes in behavior, exhibiting increased aggression, impulsivity, and a distinct deviation from the Jedi Code. Preliminary examinations reveal a heightened midichlorian count and an unsettling connection to the dark side of the Force. Further analysis is warranted to explore the extent of exposure to Sith teachings. It is imperative to monitor the individual closely for any worsening symptoms and to engage in therapeutic interventions aimed at preventing further descent into the dark side. Follow-up assessments will be crucial in determining the efficacy of intervention strategies and the overall trajectory of the individual's alignment with the Force.",
#         "dark_side_dx": "positive",
#     },
#     {
#         "patient_id_number": 2,
#         "text": "Patient exhibits no signs of succumbing to the dark side. Preliminary assessments indicate a stable midichlorian count and a continued commitment to Jedi teachings. No deviations from the Jedi Code or indicators of dark side influence were observed. Regular check-ins with the Jedi Council will ensure the sustained well-being and alignment of the individual within the Jedi Order.",
#         "dark_side_dx": "negative",
#     },
#     {
#         "patient_id_number": 3,
#         "text": "The individual manifested heightened aggression, impulsivity, and a palpable deviation from established ethical codes. Initial examinations disclosed an elevated midichlorian count and an unmistakable connection to the dark side of the Force. Further investigation is imperative to ascertain the depth of exposure to Sith doctrines. Close monitoring is essential to track any exacerbation of symptoms, and therapeutic interventions are advised to forestall a deeper embrace of the dark side. Subsequent evaluations will be pivotal in gauging the effectiveness of interventions and the overall trajectory of the individual's allegiance to the Force.",
#         "dark_side_dx": "positive",
#     }
# ])


#repo_path = Path().resolve()
output_dir = '../data/imdb/output/'
#output_dir = os.path.join(repo_path, output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv("../data/imdb/IMDB Dataset.csv")
df['id'] = range(len(df))

# register a dataset
dataset = data_handlers.from_pandas(df)

# initialize a pipeline
pipeline = Pipeline(dataset=dataset, write_path=output_dir)

# grabbing a pre-made regex set for normalizing pathology reports
pathology_regex_set = PathologyReportRegexSet().get_regex_set()

# adding the normalizer step to the pipeline
pipeline.add_step(nlp_engineering.CPUNormalizer(fields=['review'],
                                                regex_set=pathology_regex_set,
                                                lowercase=True))

# adding the pre-tokenizer step to the pipeline
pipeline.add_step(nlp_engineering.CPUPreTokenizer(fields=['review'],
                                                  split_pattern=' '))

# adding the embedding generator step to the pipeline
pipeline.add_step(nlp_engineering.CPUEmbeddingGenerator(fields=['review'],
                                                        min_word_count=2))

# adding the post processor step to the pipeline
pipeline.add_step(nlp_engineering.CPUVocabEncoder(fields=['review']))

# adding the label processor step to the pipeline
pipeline.add_step(nlp_engineering.CPULabelProcessor(fields=['sentiment']))

pipeline.add_step(nlp_engineering.CPUSplitter(
    split_method=NewSplit(
        split_proportions={"train":0.7, "test": 0.15, "val":0.15},
        unique_record_cols=["id"],
        group_cols=["id"],
        label_cols=["sentiment"],
        random_seed=42
    )
))


# run the pipeline
pipeline.run_pipeline()


pipeline_params = pipeline.get_parameters(condensed=True)

with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(pipeline_params, f, indent=4)

# grabbing the data
final_data = pipeline.processed_data.to_pandas()

# grabbing the artifacts
vocab = pipeline.artifacts['id_to_token']
label_map = pipeline.artifacts['id_to_label']
word_embeddings = pipeline.artifacts['embedding_matrix']




# print(final_data)
# print(vocab)
# print(label_map)
# print(word_embeddings)

# reviewing the collected metadata
metadata = pipeline.get_parameters()

# print(metadata)

