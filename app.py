import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
import psycopg2

app = FastAPI()
db_connect = create_engine('postgresql://awjzgmwqiatzjg:e4424ae3d375e2057bcc9cde832672940d44ea2c05260e28ccb04dc1575ec52d@ec2-34-204-22-76.compute-1.amazonaws.com:5432/dabbhqt4pegslv')
conn = db_connect.connect()

class Pointdata(BaseModel):
    device:str
    counts:int
    location:str
    vehicletype:str

@app.post("/traffic/")
async def create_data(item:Pointdata):
    try:
        query = conn.execute("insert into Traffic(deviceName,location,vehicletype,vehiclecount) values('{0}','{1}','{2}','{3}')".format(item.device,item.location,item.vehicletype,item.counts))
        return {"message":"success"}
    except:
        return {"Error":"data not sent"}
if __name__ == "__main__":

    uvicorn.run("app:app",reload=True, access_log=False)
