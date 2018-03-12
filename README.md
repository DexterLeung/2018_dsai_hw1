# 2018_dsai_hw1
DSAI HW1 - Auto Trading

<strong>Student ID</strong>: P76065013
<strong>Student Name</strong>: LEUNG Yin Chung 梁彥聰

Idea written in <code>trader.ipynb</code>:
https://nbviewer.jupyter.org/github/DexterLeung/2018_dsai_hw1/blob/master/trader.ipynb

<hr>
<h2 style="color: #33aa99; text-decoration: underline">I. Introduction</h2>
<p>
    The aim is to <strong style="color: #33aa99">predict an action</strong> for each incoming feeding stock record. The model here is to first <strong style="color: #33aa99">predict a situation</strong> from information of <em>previous 14 days</em>, followed by the action <strong style="color: #33aa99">deduced from the predicted situation</strong>.
</p><p>
    The reason is based on grabbing on any foreseeable <strong style="color: #33aa99">short interval revenues</strong> in order to get a greater return from the conservative "Buy and Hold" strategy.
</p><p>
    However, the currently implemented model is not very reliable with <strong style="color: #ee2822">varying results</strong> according to different trial data. This is probably based on some random factors in stock markets, while no generalized rules can be assumed. Since accuracy on different situation predictions may have incremental effects, the action given would also lead to a varying return result.
</p>
