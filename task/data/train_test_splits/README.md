<p># train_test_split</p>
<table>
<tbody>
<tr>
<td><strong>Woche</strong></td>
<td style="text-align: center;"><strong>von-bis</strong></td>
</tr>
<tr>
<td>26</td>
<td>23.06. - 29.06.</td>
</tr>
<tr>
<td>25</td>
<td>16.06. - 22.06.</td>
</tr>
<tr>
<td>24</td>
<td>09.06. - 15.06.</td>
</tr>
<tr>
<td>23</td>
<td>02.06. - 08.06.</td>
</tr>
<tr>
<td>22</td>
<td>26.05. - 01.06.</td>
</tr>
<tr>
<td>21</td>
<td>19.05. - 25.05.</td>
</tr>
<tr>
<td>20</td>
<td>12.05. - 18.05.</td>
</tr>
<tr>
<td>19</td>
<td>05.05. - 11.05.</td>
</tr>
<tr>
<td>18</td>
<td>28.04. - 04.05.</td>
</tr>
<tr>
<td>17</td>
<td>21.04. - 27.04.</td>
</tr>
<tr>
<td>16</td>
<td>14.04. - 20.04.</td>
</tr>
<tr>
<td>15</td>
<td>07.04. - 13.04.</td>
</tr>
<tr>
<td>14</td>
<td>31.03. - 06.04.</td>
</tr>
<tr>
<td>13</td>
<td>24.03. - 30.03.</td>
</tr>
<tr>
<td>12</td>
<td>17.03. - 23.03.&nbsp;</td>
</tr>
<tr>
<td>11</td>
<td>10.03. - 16.03.</td>
</tr>
<tr>
<td>10</td>
<td>03.03. - 09.03.</td>
</tr>
<tr>
<td>9</td>
<td>24.02. - 02.03.</td>
</tr>
<tr>
<td>8</td>
<td>17.02. - 23.02.</td>
</tr>
<tr>
<td>7</td>
<td>10.02. - 16.02.</td>
</tr>
<tr>
<td>6</td>
<td>03.02. - 09.02.</td>
</tr>
<tr>
<td>5</td>
<td>27.01. - 02.02.</td>
</tr>
<tr>
<td>4</td>
<td>20.01. - 26.01.</td>
</tr>
<tr>
<td>3</td>
<td>13.01. - 19.01.</td>
</tr>
<tr>
<td>2</td>
<td>06.01. - 12.01.</td>
</tr>
<tr>
<td>1</td>
<td>01.01. - 05.01.</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>
<table>
<tbody>
<tr>
<td><strong>Nr</strong></td>
<td><strong>Split (Woche)</strong></td>
<td><strong>Gruppe</strong></td>
<td><strong>Bemerkung</strong></td>
</tr>
<tr>
<td>1</td>
<td>25,26 Test; Rest Training</td>
<td>A</td>
<td>intuitiver Split</td>
</tr>
<tr>
<td>2</td>
<td>24,25 Test; 1-23 Training</td>
<td>A</td>
<td>intuitiver Split</td>
</tr>
<tr>
<td>3</td>
<td>23,24 Test; 1-22 Training</td>
<td>A</td>
<td>intuitiver Split</td>
</tr>
<tr>
<td>4</td>
<td>22,23 Test; 1-21 Training</td>
<td>A</td>
<td>intuitiver Split</td>
</tr>
<tr>
<td>5</td>
<td>21,22 Test; 1-20 Training</td>
<td>A</td>
<td>intuitiver Split</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>
<table>
<tbody>
<tr>
<td><strong>Score Variable</strong></td>
<td><strong>Berechung</strong></td>
</tr>
<tr>
<td><strong>Score- 1 order</strong></td>
<td>Vorhersage von einer einer Einheit</td>
</tr>
<tr>
<td><strong>Score median</strong></td>
<td>Median aller Order pro Tag pro Item&nbsp; x Anzahl an Vorhersagetagen&nbsp;</td>
</tr>
<tr>
<td>
<p><strong>Score</strong></p>
<p><strong>mean_round</strong></p>
</td>
<td>Mittelwert aller Order pro Tag pro Item&nbsp; x Anzahl an Vorhersagetagen</td>
</tr>
<tr>
<td>
<p><strong>maxScore</strong></p>
</td>
<td>Maximaler erzielbarer Score f&uuml;r Testzeitreum</td>
</tr>
</tbody>
</table>

<table>
<tbody>
<tr>
<td><strong>Datei</strong></td>
<td><strong>Beschreibung</strong></td>
</tr>
<tr>
<td>training_XX.csv</td>
<td>raw_daily.csv ohne Testzeitraum</td>
</tr>
<tr>
<td>test_XX.csv</td>
<td>itemID | simulationPrice - f&uuml;r Testzeitraum (promotion fehlt noch</td>
</tr>
<tr>
<td>test_XX_demand.csv</td>
<td>itemID | order - f&uuml;r Testzeitraum</td>
</tr>
<tr>
<td>test_XX_dates.csv</td>
<td>Datumsangaben - f&uuml;r Testzeitraum</td>
</tr>
</tbody>
</table>
