# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generates demo data for the TensorBoard projector plugin.

For a more complete walkthrough, please see one of the following tutorials:

- https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin
- https://www.tensorflow.org/tutorials/text/word_embeddings
"""

import os
import io
import numpy as np

from absl import app
from absl import flags
from tensorboard.plugins import projector

flags.DEFINE_string(
    "logdir", "/tmp/projector_demo", "Directory to write data to."
)       # /home/maria
FLAGS = flags.FLAGS

def tensor_for_label(embedding):
    """ Transform the embedding list to a tuple"""
    return tuple(embedding)


def load_vec(emb_path, nmax=20000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    print(embeddings)
    return embeddings, id2word, word2id


def write_embedding(log_dir):

    """Writes embedding data and projector configuration to the logdir."""
    nmax = 20000  # maximum number of word embeddings to load

    # Load the embeddings in each language
    english_path = r'/home/maria/projector_demo/wiki.multi.en.vec.txt'
    spanish_path = r'/home/maria/projector_demo/wiki.multi.es.vec.txt'
    french_path = r'/home/maria/projector_demo/wiki.multi.fr.vec.txt'

    english_embeddings, english_id2word, english_word2id = load_vec(english_path, nmax)
    spanish_embeddings, spanish_id2word, spanish_word2id = load_vec(spanish_path, nmax)
    french_embeddings, french_id2word, french_word2id = load_vec(french_path, nmax)

    # Add an ending mark for each language ( _en for English embeddings, _es for Spanish ones and _fr for french ones).
    # The reason to do this is that some words appeared in several languages and with this it clears to wich language the word belongs
    # Get the vocabulary of each language as a list
    english_vocab = [element[1] + "_en" for element in english_id2word.items()]
    spanish_vocab = [element[1] + "_es" for element in spanish_id2word.items()]
    french_vocab = [element[1] + "_fr" for element in french_id2word.items()]

    # Get the tensors of each language as a list
    english_tensors = [tensor_for_label(embedding) for embedding in english_embeddings]
    spanish_tensors = [tensor_for_label(embedding) for embedding in spanish_embeddings]
    french_tensors = [tensor_for_label(embedding) for embedding in french_embeddings]

    english_dic = dict(zip(english_vocab, english_tensors))
    spanish_dic = dict(zip(spanish_vocab, spanish_tensors))
    french_dic = dict(zip(french_vocab, french_tensors))

    # Concatenate the 3 dictionaries in only one and rename it
    spanish_dic.update(english_dic)
    spanish_dic.update(french_dic)

    final_dictionary = spanish_dic

    metadata_filename = "metadata.tsv"
    tensor_filename = "tensor.tsv"

    os.makedirs(log_dir, exist_ok=True)  # this methods will create a directory recursively


    # Create metadata file with 2 columns, the word and its language.
    with open(os.path.join(log_dir, metadata_filename), "w", encoding="utf-8") as f:
        f.write(f"word\tlanguage\n")
        for i, label in enumerate(final_dictionary):
            if i < nmax:
                f.write(f"{label}\tspanish\n")
            elif nmax <= i < nmax*2:
                f.write(f"{label}\tenglish\n")
            else:
                f.write(f"{label}\tfrench\n")


    with open(os.path.join(log_dir, tensor_filename), "w", encoding="utf-8") as f:
        for tensor in spanish_dic.values():
            f.write("{}\n".format("\t".join(str(x) for x in tensor)))

    # Create a config object to write the configuration parameters
    config = projector.ProjectorConfig()  # We need to tell tensorflow where it can find the metadata for the embedding data we saves to the disk.
    embedding = config.embeddings.add()  # Add embedding variable
    embedding.tensor_path = tensor_filename  # Tell the name of the variable we are interestede in visualizing
    # Link this tensor to its metadata file (e.g. labels) -> we will create this file later
    embedding.metadata_path = metadata_filename  # Tell where it can find the metadata corresponding to that variable
    # Write a projector_config.pbtxt in the logs_path.
    # TensorBoard will read this file during startup.
    projector.visualize_embeddings(log_dir, config)  #


def main(unused_argv):
    print("Saving output to %s." % FLAGS.logdir)
    write_embedding(FLAGS.logdir)
    print("Done. Output saved to %s." % FLAGS.logdir)


if __name__ == "__main__":
    app.run(main)
