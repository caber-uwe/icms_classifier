#from audioop import cross
from flask import Flask, request
from flask_cors import CORS, cross_origin
import joblib
import ICMS_ML
import re,os

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*", "methods": "GET,HEAD,PUT,PATCH,POST,DELETE", "allowedHeaders": ["Content-Type"]}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response

def checkCode(c):
    '''check if the ICMS code is valid'''
    f = re.compile(r'^[123]\.0\d\.[01]\d0$')
    if len(f.findall(c)) == 0:
        code = c.split(".")
        if len(code)!=3:
            return False
        else:
            f = re.compile(r'^[123]$')
            if len(f.findall(code[0])) == 0:
                return False
            f = re.compile(r'^0\d$')
            if len(f.findall(code[1])) == 0:
                try:
                    tmp = int(code[1])
                except:
                    return False
                if tmp < 10:
                    code[1] = "0"+str(tmp)
                else:
                    return False
            f = re.compile(r'^[01]\d0$')
            if len(f.findall(code[2])) == 0:
                try:
                    tmp = int(code[2])
                except:
                    return False
                if tmp > 190:
                    return False
                elif tmp <10:
                    code[2] = "0"+str(tmp)+'0'                    
                elif tmp%10 != 0:
                    return False
                else:
                    code[2] = "0"+str(tmp)                    

            c = '.'.join(code)
            return c
    else:
        return c
    

@app.route('/', methods=['OPTIONS','GET'])
def test():
	return {'success' : 'true', 'Response' : 'api working correctly'}

if 'clf' not in locals(): #only load the classifier once
	clf = joblib.load('Random_Forests_trained_model.joblib.gz')

icms = joblib.load('icms1_dictionnary.joblib') 
tokenList = ['a701165f12b9e3e2ebfef521fc2835b111b2fc21','1f686d8d973c21affc9b37011978c582f7885881']

@app.route('/desc/<code>', methods=['OPTIONS','GET'])
def get_desc_from_code_get(code):
	'''Get the ICMS description from the code. The code must be in the correct format'''
	f = re.compile(r'^[123]\.0\d\.[01]\d0$')
	codes = [code]
	
	descriptions = []
	for c in codes:
		code = checkCode(c)
		if code == False:
			descriptions.append({'success' : 'false', 'Response code' : '204','text':c})
			continue
		try:
			desc = icms[code]
		except:
			descriptions.append({'success' : 'false', 'Response code' : '204','text':c})
			continue

		A = {'success' : 'true','ICMS' : code ,"Description" : desc}
		R = A['ICMS'].split('.')
		D = [x.strip() for x  in A['Description'].split('\\')]
		A['R2'] = R[0]
		A['R3'] = R[1]
		A['R4'] = R[2]
		A['Desc2'] = D[0]
		A['Desc3'] = D[1]
		A['Desc4'] = D[2]
		descriptions.append(A)
	return {'descriptions':descriptions}



@app.route('/desc/', methods=['OPTIONS','POST'])
def get_desc_from_code():
	'''Get the ICMS description from the code. The code must be in the correct format'''
	f = re.compile(r'^[123]\.0\d\.[01]\d0$')
	JSON = request.get_json()
	if JSON is not None and JSON['apikey'] not in  tokenList:
		return {'success' : 'false', 'error':'invalid token'}
	codes = JSON['code']
	#if there is just one code and the list is missing
	if isinstance(codes, str):
		codes = [codes]

	descriptions = []
	for c in codes:
		code = checkCode(c)
		if code == False:
			descriptions.append({'success' : 'false', 'Response code' : '204','text':c})
			continue
		try:
			desc = icms[code]
		except:
			descriptions.append({'success' : 'false', 'Response code' : '204','text':c})
			continue

		A = {'success' : 'true','ICMS' : code ,"Description" : desc}
		R = A['ICMS'].split('.')
		D = [x.strip() for x  in A['Description'].split('\\')]
		A['R2'] = R[0]
		A['R3'] = R[1]
		A['R4'] = R[2]
		A['Desc2'] = D[0]
		A['Desc3'] = D[1]
		A['Desc4'] = D[2]
		descriptions.append(A)
	return {'descriptions':descriptions}


@app.route('/icms/', methods=['OPTIONS', 'POST'])
@cross_origin()
def classify():
	print(request.json)
	JSON = request.get_json()
	if JSON is not None and JSON['apikey'] not in  tokenList:
		return {'success' : 'false', 'Response' : 'invalid token'}
    
	print(JSON['apikey'])
	names = JSON['name']
	if isinstance(names, str):
		names = [JSON['name']]
	if len(names) > 1000:
		return {'success' : 'false', 'Response' : 'max number of codes at a time is 1000'}
	response = {'classification':ICMS_ML.predict(names,clf)}
	return response


#if os.environ.get('APP_LOCATION') == 'heroku':
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    #app.run(threaded=True, port=5000)
	#app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
#else:
    app.run(host='localhost', port=8000, debug=True)
