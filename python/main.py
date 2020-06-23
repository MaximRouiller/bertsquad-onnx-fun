# preprocess input
import run_onnx_squad as run_onnx_squad
import tokenization as tokenization
import onnxruntime
import onnx
import numpy
from onnx import numpy_helper



predict_file = 'inputs.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = run_onnx_squad.read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = './vocab.txt'
print('Running tokenizer...')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

print('Converting examples to features...')
# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
input_ids, input_mask, segment_ids, extra_data = run_onnx_squad.convert_examples_to_features(eval_examples, tokenizer,
                                                                              max_seq_length, doc_stride, max_query_length)

print('Results')
print('------------------------------------------------------')

for qa in extra_data :    
    print( " ".join(qa.tokens))