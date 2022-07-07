# UCCR
This is our implementation of the paper in SIGIR2022:

> **User-Centric Conversational Recommendation with Multi-Aspect User Modeling**
> 
> Shuokai Li, Ruobing Xie, Yongchun Zhu, Xiang Ao, Fuzhen Zhuang and Qing He
> 
> *Proceedings of the 45nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2022)*
> 
> Preprint: https://arxiv.org/abs/2204.09263

# Requirements
Please see requirements.txt.

# Run
Please run the code via:

`python run_crslab.py --config config/crs/uccr/tgredial.yaml --save_system -g 5 -m uccr_results`

The code is based on CRSLab. Thanks a lot for [CRSLab](https://github.com/RUCAIBox/CRSLab)!

# Citation

    @inproceedings{li2022user,
        title = "User-Centric Conversational Recommendation with Multi-Aspect User Modeling",
        author = "Li, Shuokai and
          Xie, Ruobing and
          Zhu, Yongchun  and
          Ao, Xiang and
          Zhuang, Fuzhen and
          He, Qing",
        booktitle = "Proceedings of the 45nd International ACM SIGIR Conference on Research and Development in Information Retrieval",
        month = July,
        year = "2022",
        publisher = "Association for Computing Machinery"
    }


