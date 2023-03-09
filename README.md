# AL4MyoChallenge
This is the solution to the baoding balls task for the [MyoChallenge2022](https://sites.google.com/view/myochallenge) by the team AL4Muscles!
<p align="center">
<img src=https://user-images.githubusercontent.com/24903880/223700843-cfa037ef-8012-4b25-a70e-e40191277212.gif width=250>
</p>
Members:

* Pierre Schumacher
* Georg Martius
* Daniel Häufle
## Installation
To run it, first install the environment

```
git clone https://github.com/martius-lab/AL4MyoChallenge
cd AL4MyoChallenge
poetry install
poetry shell
```
then run

`python -m al4myochallenge.main baoding_verification.json`
## Performance
We achieved 98% in phase 1 and 41% in phase 2, which landed us the second place in the baoding balls task.
<p align="center">
<img src=https://user-images.githubusercontent.com/24903880/223505877-d610a8fd-4cdc-4acc-8c12-21cf30681f28.png width=200> 
</p>

The above plot shows the score in the evaluation environment over training for 10 random seeds.

<p align="center">
<img src=https://user-images.githubusercontent.com/24903880/223701896-4024535b-d169-43fa-a3de-1dbcdf3c77d7.png width=200><img src=https://user-images.githubusercontent.com/24903880/223701907-fed1ff3c-f940-45a2-a8e1-bc5f9d659ee9.png width=195>
</p>
