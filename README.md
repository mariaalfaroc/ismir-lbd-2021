# Multimodal image and audio music transcription

![Tensorflow](https://img.shields.io/badge/Tensorflow-%FFFFFF.svg?style=flat&logo=Tensorflow&logoColor=orange&color=white) [![License](https://img.shields.io/static/v1?label=License&message=MIT&color=blue)]() 

Code for the paper:<br />
  > Carlos de la Fuente, Jose J Valero-Mas, Francisco J Castellanos, Jorge Calvo-Zaragoza, **María Alfaro-Contreras**, and Jose M Iñesta<br />
  [*Multimodal Audio and Image Music Transcription*](https://archives.ismir.net/ismir2021/latebreaking/000022.pdf)<br />
  Late-breaking Demo at the 22nd International Society for Music Information Retrieval (ISMIR) Conference, Online, November 7-12, 2021

Dataset used: **Camera-PrIMuS**. Available [here](https://grfia.dlsi.ua.es/primus/).
The partitions used can be found in the 5-crossval.tgz file.

----

**Citation**

```bibtex
@inproceedings{delafuente2021multimodal,
  author       = {de la Fuente, Carlos and Valero-Mas, Jose J. and Castellanos, Francisco J. and Calvo-Zaragoza, Jorge and Alfaro-Contreras, Mar{\'i}a and I{\~n}esta, Jose M. },
  title        = {{Multimodal audio and image music transcription}},
  booktitle    = {{Late-breaking Demo at the 22nd International Society for Music Information Retrieval (ISMIR) Conference}},
  year         = 2021,
  month        = nov,
  address      = {Online},
}
```

----

**Requirements**

tensorflow-gpu==2.3.1<br />
pandas==1.3.0<br />
numpy==1.18.5<br />
opencv-python==4.5.3.56
swalign==0.3.6
