# EXVO

Code to run EXVO experiments.
Code is based on downloading the original data.

Then running `./melspects.py` to extract features, followed by `training.py` to run the training.

Results
====================

====================
TASK1
====================
| Model | Emo-CCC | Cou-UAR | Age-MAE | Score |
| - | - | - | - | - |
| cnn14-random-init | 0.4900 | 0.5010 | 3.9952 | 0.3735 |
| cnn10-random-init | 0.5137 | 0.4993 | 4.0379 | 0.3756 |
| cnn14-audioset | 0.4028 | 0.4400 | 4.0206 | 0.3418 |

====================
TASK3
====================
| Model | Emo-CCC | Country-mean | Country-std | Speaker-mean | Speaker-std |
| - | - | - | - | - | - |
| cnn14-audioset | 0.5648 [0.5604-0.5694] | 0.5620 | 0.0097| 0.5251 | 0.1431 |
| cnn14-random-init | 0.6393 [0.6339-0.6438] | 0.6377 | 0.0112| 0.5866 | 0.1646 |
| cnn10-random-init | 0.6270 [0.6222-0.6320] | 0.6217 | 0.0178| 0.5747 | 0.1614 |
| cnn14-random-init-personalisation | 0.6519 [0.6463-0.6570] | 0.6468 | 0.0190| 0.5998 | 0.1660 |
| cnn14-random-init-personalisation-continued | 0.6601 [0.6546-0.6651] | 0.6561 | 0.0157| 0.6101 | 0.1723 |
| cnn14-random-init-personalisation-multitasking | 0.6343 [0.6286-0.6396] | 0.6303 | 0.0147| 0.5814 | 0.1631 |
| cnn14-random-init-personalisation-multitasking-continued | 0.6563 [0.6506-0.6615] | 0.6520 | 0.0157| 0.6051 | 0.1661 |
| cnn14-random-init-continued | 0.6429 [0.6374-0.6477] | 0.6395 | 0.0151| 0.5909 | 0.1671 |
| cnn14-attention-cnn10-learn-auxil-forget-main | 0.6391 [0.6339-0.6438] | 0.6357 | 0.0122| 0.5859 | 0.1609 |
| cnn14-attention-cnn10-multitask-adversarial | 0.6601 [0.6546-0.6649] | 0.6556 | 0.0161| 0.6073 | 0.1706 |
| cnn14-attention-cnn10-multitask | 0.6572 [0.6518-0.6621] | 0.6533 | 0.0136| 0.6043 | 0.1677 |


====================
TASK3
====================
| Model | Emo-CCC | Country-mean | Country-std | Speaker-mean | Speaker-std |
| - | - | - | - | - | - |
| cnn14-audioset | 0.5648 | 0.5620 | 0.0097| 0.5251 | 0.1431 |
| cnn14-random-init | 0.6393 | 0.6377 | 0.0112| 0.5866 | 0.1646 |
| cnn10-random-init | 0.6270 | 0.6217 | 0.0178| 0.5747 | 0.1614 |
| cnn14-random-init-personalisation | 0.6519 | 0.6468 | 0.0190| 0.5998 | 0.1660 |
| cnn14-random-init-personalisation-continued | 0.6601 | 0.6561 | 0.0157| 0.6101 | 0.1723 |
| cnn14-random-init-personalisation-multitasking | 0.6343 | 0.6303 | 0.0147| 0.5814 | 0.1631 |
| cnn14-random-init-personalisation-multitasking-continued | 0.6563 | 0.6520 | 0.0157| 0.6051 | 0.1661 |
| cnn14-random-init-continued | 0.6429 | 0.6395 | 0.0151| 0.5909 | 0.1671 |
| cnn14-attention-cnn10-learn-auxil-forget-main | 0.6391 | 0.6357 | 0.0122| 0.5859 | 0.1609 |
| cnn14-attention-cnn10-learn-auxil | 0.6572 | 0.6525 | 0.0169| 0.6074 | 0.1697 |
| cnn14-attention-cnn10-multitask-adversarial | 0.6601 | 0.6556 | 0.0161| 0.6073 | 0.1706 |
| cnn14-attention-cnn10-multitask | 0.6572 | 0.6533 | 0.0136| 0.6043 | 0.1677 |


| cnn14-bs8 | .759 |.621 |.726 |.508 |.683 |.672 |.614 |.569 |.577 |.726
| cnn14-attention-cnn10-layernorm-full | .763 |.628 |.741 |.506 |.698 |.676 |.603 |.555 |.641 |.748
| cnn14-attention-cnn10-layernorm-auxil-learn | .765 |.634 |.736 |.503 |.689 |.669 |.606 |.563 |.640 |.742
| cnn14-attention-cnn10-layernorm-auxil-learn-multitask-adversarial | .773 |.612 |.737 |.513 |.687 |.674 |.590 |.559 |.625 |.746
| cnn14-attention-cnn10-layernorm-auxil-learn-multitask-adversarial-forget-main | .740 |.600 |.736 |.480 |.686 |.675 |.592 |.505 |.604 |.738
| cnn14-speaker-multitasking | .679 |.550 |.680 |.396 |.600 |.576 |.493 |.441 |.484 |.605
| cnn14-attention-cnn10-layernorm-multitask-adversarial | .766 |.631 |.749 |.507 |.692 |.669 |.609 |.566 |.647 |.747
| cnn14-speaker-invariant | .725 |.584 |.690 |.472 |.659 |.650 |.568 |.498 |.552 |.680


| cnn14-bs8 | 0.6454 | f + g | fewshot_EIHW_0 |
| cnn14-attention-cnn10-layernorm-full | 0.6559 | f + g + tilde(g) | fewshot_EIHW_1 |
| cnn14-attention-cnn10-layernorm-auxil-learn | 0.6547 | f + g + tilde(g) + tilde(h) | fewshot_EIHW_2 |
| cnn14-attention-cnn10-layernorm-multitask-adversarial | 0.6582 |  f + g + tilde(g) - tilde(f) | fewshot_EIHW_3 |
| cnn14-attention-cnn10-layernorm-auxil-learn-multitask-adversarial | 0.6516 | f + g + tilde(h) - tilde(f) | fewshot_EIHW_4 |