from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..models.synthetic_data_models import SyntheticDataSet
from ..core.config import settings
import os
import pandas as pd
from datetime import datetime
from ..models.synthetic_data_models import DatasetRow  # We'll define this model if not present

router = APIRouter()

@router.post('/datasets/upload')
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file to /tmp for parsing
    temp_path = f"/tmp/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    with open(temp_path, 'wb') as out_file:
        out_file.write(await file.read())
    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
    os.remove(temp_path)

    # Store each row in DatasetRow table
    for _, row in df.iterrows():
        db_row = DatasetRow(**row.to_dict())
        db.add(db_row)
    db.commit()

    return {"success": True, "rows_added": len(df)} 
