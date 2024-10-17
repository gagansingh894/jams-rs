wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

wrk.body = [[
{
  "model_name": "titanic_model",
  "input": "{\"adult_male\":[\"True\",\"False\"],\"age\":[22.0,23.8],\"alone\":[\"True\",\"False\"],\"class\":[\"First\",\"Third\"],\"deck\":[\"Unknown\",\"Unknown\"],\"embark_town\":[\"Southampton\",\"Cherbourg\"],\"embarked\":[\"S\",\"C\"],\"fare\":[151.55,14.4542],\"parch\":[\"0\",\"0\"],\"pclass\":[\"1\",\"3\"],\"sex\":[\"male\",\"female\"],\"sibsp\":[\"0\",\"1\"],\"who\":[\"man\",\"woman\"]}"
}
]]