Q1,2,3,4,5:
j'ai cree le projet est le ompte service
![alt text](5.png)

Q6:
ajouter le clé.
![alt text](6.png)

Q7:
![alt text](image.png)

Q8:
![alt text](8.png)

Q9:
![alt text](9.png)

Q10:
![alt text](10.png)

Q11:
pip install -r requirements.txt
pip install dvc dvc-gdrive

Q12:
dvc init
![alt text](image-1.png)
Q13:
dvc add data 
![alt text](image-2.png)
git add data.dvc 

Q14:
![alt text](image-3.png)

Q15:
dvc remote modify gdrive_remote gdrive_use_service_account true

Q16:
dvc remote modify gdrive_remote gdrive_acknowledge_abuse true

Q17: 
dvc remote modify gdrive_remote --local gdrive_service_account_json_file_path data-management-480310-e6b0fe327b1c.json

Q18:
dvc config core.autostage true
Q19:
![alt text](image-4.png)

Q20:
dvc push
![alt text](image-5.png)
![alt text](image-6.png)
