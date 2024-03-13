import React, { useState, useEffect } from 'react';
import './Results.css'

function Results({ formSubmitted, responseData }) {
  const [plotUrls, setPlotUrls] = useState([]);

  useEffect(() => {
    async function fetchPlotUrls() {
      const urls = [];
      for (const item of responseData) {
        if (typeof item === 'string' && item.endsWith('.png')) {
          // Assuming your Flask server runs on the same domain, adjust URL accordingly if it's different
          const response = await fetch(`/get_plot/${item}`);
          if (response.ok) {
            const blob = await response.blob();
            urls.push(URL.createObjectURL(blob));
          }
        }
      }
      setPlotUrls(urls);
    }

    if (formSubmitted && responseData) {
      fetchPlotUrls();
    }
  }, [formSubmitted, responseData]);

  return (
    <div className='results-container'>
      {formSubmitted && responseData && (
        <div className=''>
          <h1 className="">Results:</h1>
          <p>1 year price history:</p>
          <img src={plotUrls[0]} alt="" className="" /><br/>
          <p className="">Splitting the training data into training set and validation set:</p>
          <img src={plotUrls[1]} alt="" className="" /><br/>
          <img src={plotUrls[2]} alt="" className="" /><br/>
          <img src={plotUrls[3]} alt="" className="" /><br/>
          <img src={plotUrls[4]} alt="" className="" /><br/>
          <p className="">{responseData[8]}</p><br/><br/>
          <hr/>
          <p className="">{responseData[10]}<br/><br/>{responseData[11]}<br/><br/>{responseData[9]}</p>
          <img src={plotUrls[5]} alt="" className="" />
          <hr/>
          <p className="">{responseData[13]}<br/><br/>{responseData[14]}<br/><br/>{responseData[12]}</p>
          <img src={plotUrls[6]} alt="" className="" />
          <hr/>
          <p className="">{responseData[16]}<br/><br/>{responseData[17]}<br/><br/>{responseData[15]}</p>
          <img src={plotUrls[7]} alt="" className="" />
        </div>
      )}
    </div>
  );
}

export default Results;
