# Varroa MMDetection Results Summary

Downloaded Hugging Face snapshots into `hf_runs/` with `*.pt` and `*.pth` ignored.

## Mean By Model/Variant

<table>
  <thead>
    <tr>
      <th>model</th>
      <th>variant</th>
      <th>n</th>
      <th>seeds</th>
      <th>mAP</th>
      <th>mAP50</th>
      <th>mAP75</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="7" style="border-top: 2px solid #000;"><strong>Base</strong></td></tr>
    <tr><td>atss</td><td>base</td><td>3</td><td>42,43,44</td><td>0.335 ± 0.010</td><td>0.907 ± 0.012</td><td>0.119 ± 0.002</td></tr>
    <tr><td>cascade_rcnn</td><td>base</td><td>3</td><td>42,43,44</td><td>0.334 ± 0.012</td><td>0.900 ± 0.010</td><td>0.121 ± 0.016</td></tr>
    <tr><td>dyhead</td><td>base</td><td>3</td><td>42,43,44</td><td>0.331 ± 0.003</td><td>0.905 ± 0.003</td><td>0.112 ± 0.006</td></tr>
    <tr><td>faster_rcnn</td><td>base</td><td>3</td><td>42,43,44</td><td>0.338 ± 0.010</td><td>0.897 ± 0.016</td><td>0.116 ± 0.012</td></tr>
    <tr><td>fcos</td><td>base</td><td>3</td><td>42,43,44</td><td>0.255 ± 0.074</td><td>0.777 ± 0.175</td><td>0.069 ± 0.034</td></tr>
    <tr><td>tood</td><td>base</td><td>3</td><td>42,43,44</td><td>0.335 ± 0.006</td><td>0.884 ± 0.009</td><td>0.125 ± 0.011</td></tr>
    <tr><td colspan="7" style="border-top: 1px solid #000;"><strong>SR-TOD(Theirs)</strong></td></tr>
    <tr><td>cascade</td><td>srtod</td><td>1</td><td>44</td><td>0.315 ± 0.000</td><td>0.877 ± 0.000</td><td>0.101 ± 0.000</td></tr>
    <tr><td>faster</td><td>srtod</td><td>1</td><td>44</td><td>0.316 ± 0.000</td><td>0.885 ± 0.000</td><td>0.106 ± 0.000</td></tr>
    <tr><td colspan="7" style="border-top: 1px solid #000;"><strong>DGFE API(Ours)</strong></td></tr>
    <tr><td>atss</td><td>dgfe_api</td><td>3</td><td>42,43,44</td><td>0.346 ± 0.002</td><td>0.898 ± 0.012</td><td>0.152 ± 0.007</td></tr>
    <tr><td>cascade_rcnn</td><td>dgfe_api</td><td>3</td><td>42,43,44</td><td>0.350 ± 0.024</td><td>0.907 ± 0.012</td><td>0.143 ± 0.044</td></tr>
    <tr><td>dyhead</td><td>dgfe_api</td><td>2</td><td>42,43</td><td>0.335 ± 0.011</td><td>0.908 ± 0.006</td><td>0.121 ± 0.009</td></tr>
    <tr><td>faster_rcnn</td><td>dgfe_api</td><td>3</td><td>42,43,44</td><td>0.353 ± 0.020</td><td>0.903 ± 0.009</td><td>0.158 ± 0.034</td></tr>
    <tr><td>fcos</td><td>dgfe_api</td><td>3</td><td>42,43,44</td><td>0.315 ± 0.008</td><td>0.879 ± 0.012</td><td>0.114 ± 0.013</td></tr>
    <tr><td>tood</td><td>dgfe_api</td><td>3</td><td>42,43,44</td><td>0.338 ± 0.006</td><td>0.874 ± 0.015</td><td>0.148 ± 0.004</td></tr>
  </tbody>
</table>
