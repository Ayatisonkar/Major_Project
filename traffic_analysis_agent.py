
from datetime import datetime
import ollama
import json
from flask import Flask, request, jsonify
# Set the base URL explicitly
ollama.Client(host='http://localhost:11434')

app = Flask(__name__)

def extract_from_first_to_last_brace(text):
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    else:
        return None
def image_to_json(data):
    response = ollama.generate(
        model='llava',
        prompt = f"""Verify the violation of traffic rules in the given image and compare it with helmet_violation = {data['helmet_violation']} and overcrowded_violation = {data['overcrowded_violation']}. Get the license plate number from the image if the driver on the motorcycle is not wearing a helmet or if the motorcycle is overcrowded, and return it in JSON format. 
     If it is violating other traffic rules, return the license plate number and the traffic rule violated in JSON format. If the lisence plate is not visible mention not visible. This is the entire data avialable to you {data} 
     The JSON must look like this:
     {{
         "timestamp": "{datetime.now().isoformat()}",
         "plate": "plate_number",
         "violations": [
             {{
                 "type": "no_helmet" if {data['helmet_violation']} is True or motorcycle_is_overcrowded if {data['overcrowded_violation']} is True or both.
                 "description": "Driver on motorcycle not wearing helmet" if {data['helmet_violation']}==True else "Motorcycle overcrowded" if {data['overcrowded_violation']}==True else "Other traffic rule violated",
             }}
         ]
     }}
     Only give json as output""",
        images=[data['context_image']],  # Load the image
    )
    output_file = "/home/eleensmathew/Traffic/output.json"
    answer = extract_from_first_to_last_brace(response['response'])
    if answer:
        # Save the JSON to a file
        with open(output_file, "a") as file:
            file.write(answer+"\n")
        print(f"JSON saved to {output_file}")
    
    return response['response']

# # result = image_to_json("/home/eleensmathew/Traffic/videos/image.png")
# # print(result)

# data = {
#     "context_image": "/home/eleensmathew/Traffic/image copy.png",  # Load the image
#     "helmet_violation": True,  # Example: Driver not wearing a helmet
#     "overcrowded_violation": True,  # Example: Motorcycle is not overcrowded
# }

# # Call the function
# result = image_to_json(data)

# # Print the result
# print(result)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    result = image_to_json(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0')