import urllib.request, json

data = {'MonsoonIntensity':8,'TopographyDrainage':5,'RiverManagement':4,'Deforestation':7,'Urbanization':8,'ClimateChange':9,'DamsQuality':3,'Siltation':6,'AgriculturalPractices':5,'Encroachments':7,'IneffectiveDisasterPreparedness':8,'DrainageSystems':3,'CoastalVulnerability':7,'Landslides':6,'Watersheds':4,'DeterioratingInfrastructure':7,'PopulationScore':8,'WetlandLoss':6,'InadequatePlanning':7,'PoliticalFactors':8}

payload = json.dumps(data).encode()
req = urllib.request.Request('http://localhost:5000/api/predict', data=payload, headers={'Content-Type':'application/json'})
r = urllib.request.urlopen(req)
result = json.loads(r.read())

print("=== PREDICTION RESULT ===")
print("Individual Models:")
for k,v in result['predictions']['individual'].items():
    print(f"  {k}: {v*100:.2f}%")
print(f"Ensemble Average: {result['predictions']['flood_probability_percent']}%")
print(f"Risk Level: {result['risk_assessment']['level']}")
print(f"Recommendations: {len(result['recommendations'])} generated")
print("\nRecommendations:")
for rec in result['recommendations']:
    print(f"  - {rec}")
