import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import asyncpg
from fastapi.middleware.cors import CORSMiddleware
from databases import Database

app = FastAPI()
database = Database('postgresql://awjzgmwqiatzjg:e4424ae3d375e2057bcc9cde832672940d44ea2c05260e28ccb04dc1575ec52d@ec2-34-204-22-76.compute-1.amazonaws.com:5432/dabbhqt4pegslv')

@app.on_event("startup")
async def database_connect():
    await database.connect()


@app.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()

class Pointdata(BaseModel):
    device:str
    counts:int
    location:str
    vehicletype:str

@app.post("/traffic/")
async def create_data(item:Pointdata):
    try:
        # query = conn.execute("insert into Traffic(deviceName,location,vehicletype,vehiclecount) values('{0}','{1}','{2}','{3}')".format(item.device,item.location,item.vehicletype,item.counts))
        await database.execute("insert into Traffic(deviceName,location,vehicletype,vehiclecount) values('{0}','{1}','{2}','{3}')".format(item.device,item.location,item.vehicletype,item.counts))
        result= {"message":"success"}
    except:
        result= {"Error":"data not sent"}
    return result

@app.post("/fetchdata/")
async def read_results():
    try:
        query = "select * from Traffic"
        result = await database.fetch_all(query=query)
    except:
        result= {"Error":"Database is empty"}
    return result
if __name__ == "__main__":

    uvicorn.run("app:app",reload=True, access_log=False)
