import parser

# constants 
DEMENTIABANK_CONTROL_DIR  = 'data/processed/dbank/control' 
DEMENTIABANK_DEMENTIA_DIR = 'data/processed/dbank/dementia'
OPTIMA_CONTROL_DIR        = 'data/processed/optima/nometa/control' 
OPTIMA_DEMENTIA_DIR       = 'data/processed/optima/nometa/dementia' 

if __name__ == '__main__':
	
	# 1. parse arguments
	# 2. Call parser to make list of strings
	# 3. Call extractor to make list of features
	# 4. Make arff file
	# 5. export / call weka to train
	
	dbank_control  = parser.parse(DEMENTIABANK_CONTROL_DIR)
	dbank_dem      = parser.parse(DEMENTIABANK_DEMENTIA_DIR)
	optima_control = parser.parse(OPTIMA_CONTROL_DIR)
	optima_dem     = parser.parse(OPTIMA_DEMENTIA_DIR)
	
	print "DBank Control: "  + str(len(dbank_control))
	print "DBank Dem: " 	 + str(len(dbank_dem))
	print "Optima Control: " + str(len(optima_control))
	print "Optima Dem: "	 + str(len(optima_dem))