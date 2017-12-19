import input_data

infile = open('output.txt', 'r')
outfile = open('postprocessed_output', 'w')

CONFIDENCE = 0.3

# fname,label
outfile.write(infile.readline())

labels_count = {}

line = infile.readline()
while line:
  wav, label, confidence, _ = line.split(',')
  if float(confidence) < CONFIDENCE:
    label = 'unknown'
  if label == input_data.UNKNOWN_WORD_LABEL:
    label = 'unknown'
  if label == input_data.SILENCE_LABEL:
    label = 'silence'
  if label in labels_count:
    labels_count[label] += 1
  else:
    labels_count[label] = 1
  outfile.write('%s,%s\n' % (wav, label))
  line = infile.readline()
  
infile.close()
outfile.close()

for label in labels_count:
  print '%15s : %d' % (label, labels_count[label])