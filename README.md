# med-table2text

### Medical multi-modal generative model
- We generate text (operation reports) from table. 
- Dataset is not opened.
- Detailed description will be updated shortly.

### Run the code below to train the model.
```python
cd commands
bash train.sh {device_num} {checkpoint_name} {mlm_ratio}
```
### Run the code below to test the model.
```python
cd commands
bash test.sh {device_num} {checkpoint_name} {mlm_ratio}
```
