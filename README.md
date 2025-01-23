# Passport detection service

![badge](https://github.com/icij/passport-service/actions/workflows/test-passport-service.yml/badge.svg)

## Usage

### Run the service

Since some dependencies of the service are platform dependent (ARM64 vs X86_64), a small wrapper around the
`docker compose` command was created to run the service.
You can use it just like `docker compose`:

```bash
./passport-service up -d
```

or

```bash
./passport-service up
```

<br>

### API documentation

Once the service is up the API documentation is available at [http://localhost:8080/docs](http://localhost:8080/docs).

<br>

### Detect passports

#### Model

Place your YOLO model exported as a `.onnx` file in the [data/models](data/models) directory.

Make sure it's correctly exported for batch processing (see how to do
it [here](#note-on-onnx-export-and-dynamic-batch-size)).

#### Data

To detect passport you first need to place your documents inside the [data/passports](data/passports) directory.

#### Preprocessing

To process passport run:

```console
curl -X 'POST' \
  'http://localhost:8080/passports/preprocessing-tasks' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "docs": "my_doc_dir",
  "detection_args": {"model_path": "model_v0.onnx"},
  "batch_size": 64
}'
```

This will return you something like:

```console
create-preprocessing-tasks-fd05a3b4d2774b269001cccce2fcf073
```

You can follow the task progress running:
```
curl -X 'GET' 'http://localhost:8080/tasks/create-preprocessing-tasks-fd05a3b4d2774b269001cccce2fcf073' \
  -H 'accept: application/json'
```
and read the progress from the response.

Then when it's done, you can get the list of preprocessing tasks you can call:

```console
curl -X 'GET' 'http://localhost:8080/tasks/create-preprocessing-tasks-fd05a3b4d2774b269001cccce2fcf073/result' \
  -H 'accept: application/json'
```

Similarly, you can follow each preprocessing task progress.


Preprocessing task will then trigger passport detection on the output of the preprocessing.

You can get the ID of the created detection task from the preprocessing task result:

```console
curl -X 'GET' 'http://localhost:8080/tasks/preprocess-docss-090943423909090000/result' \
  -H 'accept: application/json'
```

When the deteection task is done, you can also call the above API with it ID to get the final output which should look like this:

```json
[
  {
    "doc_path": "passports/passport.odt",
    "doc_pages": [
      {
        "page": 0,
        "passports": [
          {
            "class_id": "passport",
            "confidence": 0.9391622543334961,
            "box": [
              57.185882568359375,
              216.44244384765625,
              375.8140869140625,
              283.35162353515625
            ],
            "scale": 1.2375,
            "mrz": {
              "country": "EOL",
              "metadata": {
                "mrz_type": "TD3",
                "valid_score": 62,
                "raw_text": "P<EOLSMITH<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n01PP300009EOL8107145F2212315<<<<<<<<<<<<<<02",
                "type": "P<",
                "country": "EOL",
                "number": "01PP30000",
                "date_of_birth": "810714",
                "expiration_date": "221231",
                "nationality": "EOL",
                "sex": "F",
                "names": "JANE",
                "surname": "SMITH",
                "personal_number": "<<<<<<<<<<<<<<",
                "check_number": "9",
                "check_date_of_birth": "5",
                "check_expiration_date": "5",
                "check_composite": "2",
                "check_personal_number": "0",
                "valid_number": false,
                "valid_date_of_birth": true,
                "valid_expiration_date": true,
                "valid_composite": false,
                "valid_personal_number": true,
                "method": "direct"
              }
            }
          },
          {
            "class_id": "passport",
            "confidence": 0.9231931567192078,
            "box": [
              58.3187255859375,
              54.883644104003906,
              373.454345703125,
              185.15440368652344
            ],
            "scale": 1.2375,
            "mrz": null
          }
        ]
      }
    ]
  },
  {
    "doc_path": "passports/not_a_passport.jpg",
    "doc_pages": []
  }
]
```

Notice that for each document, only pages with passports inside them are reported.
On these page, there can be many passport pages, the algorithm outputs a bounding box and optionally a Machine Readable
Zone (MRZ) for each one of these pages.

<br>

## Coming soon...

- pipelined pre-processing and detection

<br>

## Note on ONNX export and dynamic batch size

To overcome some OpenCV limitations, export the pytorch checkpoint using:

```python
model.export(format='onnx', imgsz=640, dynamic=True, opset=12, verbose=True, simplify=True)
```

Then, follow the instructions of [this issue](https://github.com/opencv/opencv/issues/25485) to make OpenCV work with
dynamic batch size.

1. simplify using `onnxsim`
2. set the `height` and `width` using:

```bash
python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param <param> --dim_value 640 <input> <output>
```