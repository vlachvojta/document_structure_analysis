# readme

Part of tables for initial parsing and stuff

XML sub-table elements:
```xml
...
    <object>
      <name>table</name>
      <pose>Frontal</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <occluded>0</occluded>
      <bndbox>
         <xmin>36.5556</xmin>
         <ymin>36.6549</ymin>
         <xmax>646.0124</xmax>
         <ymax>284.3667</ymax>
      </bndbox>
   </object>
...
```

Words JSONS:
```json
[
  {
    "bbox": [
      31.47979797979798,
      0.9930303030303094,
      95.38327020202019,
      12.852196969696962
    ],
    "text": "Comparisons",
    "flags": 0,
    "span_num": 515,
    "line_num": 0,
    "block_num": 0
  },
  {
    "bbox": [
      98.13597222222222,
      0.9930303030303094,
      108.02532828282827,
      12.852196969696962
    ],
    "text": "of",
    "flags": 0,
    "span_num": 516,
    "line_num": 0,
    "block_num": 0
  },
]
```